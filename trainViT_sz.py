"""ViT ansatz for the XY model on fractal lattices."""


import argparse
import json
import os
import time
from functools import partial
from math import comb

import flax
import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
from jax.flatten_util import ravel_pytree
from tqdm import tqdm

from sector_fidelity import (N_PATCH, RESTORED_DISCARD, _base, exact_energy, exact_vector, load_sector_state,
                             load_variables, project_c3v, reference_energy, sector_mcstate, target_sz)
from trainViT import build_fractal, patch_distances
from utils import build_hamiltonian, save_metadata, save_vstate

# Sectors with at most this many states are trained with exact sums. Kept apart from
# sector_fidelity.FULLSUM_MAX, which also sums triangular G=2 exactly when evaluating
FULLSUM_MAX = 1000

STAGES = ("main", "c3v")


def exact_sr_trainer(vstate, hi, H, diag_shift):
    """Jitted SR step that sums over every configuration of a small sector.

    With real parameters theta, complex log psi and p = |psi|^2 / <psi|psi>:
      X  = [Re; Im] of sqrt(p) (O - <O>),       O_k = d log psi / d theta_k   (2D x P)
      e  = [Re; Im] of sqrt(p) (E_loc - E)                                     (2D)
      dtheta = -lr X^T (X X^T + diag_shift)^-1 e
    which equals -lr S^-1 F with S = X^T X, F = X^T e, the usual SR update, but
    only needs a 2D x 2D solve (D = number of states).
    """
    sigma = jnp.asarray(hi.all_states())
    H_dense = jnp.asarray(H.to_dense())
    variables = flax.core.unfreeze(vstate.variables)
    flat0, unravel = ravel_pytree(variables["params"])
    rest = {k: v for k, v in variables.items() if k != "params"}
    model = vstate.model

    def log_psi(flat, s):
        return model.apply({**rest, "params": unravel(flat)}, s)

    def single(flat, s):
        return log_psi(flat, s[None])[0]

    grad_re = jax.vmap(jax.grad(lambda f, s: single(f, s).real), in_axes=(None, 0))
    grad_im = jax.vmap(jax.grad(lambda f, s: single(f, s).imag), in_axes=(None, 0))

    def state(flat):
        lp = log_psi(flat, sigma)
        psi = jnp.exp(lp - lp.real.max())
        psi = psi / jnp.linalg.norm(psi)
        Hpsi = H_dense @ psi
        E = jnp.vdot(psi, Hpsi).real
        var = jnp.vdot(Hpsi, Hpsi).real - E**2
        return psi, Hpsi, E, var

    @jax.jit
    def step(flat, lr):
        psi, Hpsi, E, var = state(flat)
        amp = jnp.abs(psi)
        phase = jnp.where(amp > 0, psi.conj() / jnp.where(amp > 0, amp, 1.0), 0.0)
        eps = Hpsi * phase - amp * E                    # sqrt(p) (E_loc - E), safe where psi = 0
        O = grad_re(flat, sigma) + 1j * grad_im(flat, sigma)
        O = amp[:, None] * (O - (amp**2) @ O)
        X = jnp.concatenate([O.real, O.imag])
        e = jnp.concatenate([eps.real, eps.imag])
        K = X @ X.T + diag_shift * jnp.eye(X.shape[0])
        dtheta = X.T @ jax.scipy.linalg.solve(K, e, assume_a="pos")
        return flat - lr * dtheta, E, var

    def write_back(flat):
        vstate.variables = {**rest, "params": unravel(flat)}

    return step, flat0, write_back


@partial(jax.jit, static_argnames=("log_psi", "solver_fn", "chunk_size", "n_blocks"))
def blocked_srt(log_psi, local_energies, parameters, model_state, samples, *, diag_shift, solver_fn,
                chunk_size, n_blocks, proj_reg=None, momentum=None, old_updates=None):
    """The update of NetKet's VMC_SR(use_ntk=True, mode="complex") for real parameters,
    holding at most one column block of the Jacobian, and no copy of it.

    With J the Jacobian of [Re; Im] log psi (a Re and an Im row per sample), P the centring
    over samples and O = P J / sqrt(n_samples), as in NetKet:
      T = O O^T = P (sum_k J_k J_k^T) P / n_samples     over n_blocks column blocks J_k
      a = (T + diag_shift)^-1 (dv - momentum O old_updates)
      updates = O^T a + momentum old_updates
    With one block, O^T a uses the Jacobian in memory; with more, it is a chunked vjp. The rows
    are built shifted by the mean Jacobian of 32 samples, which P removes again, so that the
    centring of T does not cancel a large common part of them (as in NetKet >= 3.22).
    Returns (updates, flat updates for the momentum of the next step).
    """
    n = samples.shape[0]
    theta, unravel = ravel_pytree(parameters)
    n_params = theta.size
    width = -(-n_params // n_blocks)
    n_blocks = -(-n_params // width)
    chunks = samples.reshape(n // chunk_size, chunk_size, -1)

    def re_im(th, x):
        lp = log_psi({**model_state, "params": unravel(th)}, x)
        return jnp.stack([lp.real, lp.imag], axis=-1)

    jac = jax.vmap(jax.jacrev(lambda th, s: re_im(th, s[None])[0]), in_axes=(None, 0))
    shift = jac(theta, samples[::max(1, n // min(32, chunk_size))]).mean(axis=0)

    def columns(start, keep=1.0):
        """Columns start:start + width of the shifted Jacobian, (2 n_samples, width)"""
        cols = lambda J: jax.lax.dynamic_slice_in_dim(J, start, width, J.ndim - 1)
        return jax.lax.map(lambda x: (cols(jac(theta, x)) - cols(shift)) * keep, chunks).reshape(2 * n, width)

    # Products with a block are taken over pieces of at most 2 n_samples of its columns: the GPU
    # compiler times every matrix product on fresh copies of its operands, which for J_k J_k^T
    # of a whole block needs twice its memory (out of memory at G=2 with one 12.4 GiB block)
    piece = min(width, 2 * n)

    def gram(J, u, T, Ju):
        """T + J J^T and Ju + J u, over pieces of J"""
        def add(k, carry):
            T, Ju = carry
            # the last piece is shifted back to fit; its columns that belong to piece k-1 are zeroed
            start = jnp.minimum(k * piece, width - piece)
            Jp = jax.lax.dynamic_slice_in_dim(J, start, piece, 1) * (start + jnp.arange(piece) >= k * piece)
            if momentum is not None:
                Ju = Ju + Jp @ jax.lax.dynamic_slice_in_dim(u, start, piece)
            return T + Jp @ Jp.T, Ju

        return jax.lax.fori_loop(0, -(-width // piece), add, (T, Ju))

    zeros = jnp.zeros((2 * n, 2 * n)), jnp.zeros(2 * n)
    if n_blocks == 1:
        J = columns(0)
        T, Ju = gram(J, old_updates, *zeros)
    else:
        def add_block(k, carry):
            # the last block is shifted back to fit; its columns that belong to block k-1 are zeroed
            start = jnp.minimum(k * width, n_params - width)
            Jk = columns(start, start + jnp.arange(width) >= k * width)
            u = jax.lax.dynamic_slice_in_dim(old_updates, start, width) if momentum is not None else None
            return gram(Jk, u, *carry)

        T, Ju = jax.lax.fori_loop(0, n_blocks, add_block, zeros)

    def centre(v):
        """P on vectors of 2 n_samples entries, divided by sqrt(n_samples)"""
        v = v.reshape(n, 2)
        return ((v - v.mean(axis=0)) / n**0.5).reshape(-1)

    T = T.reshape(n, 2, n, 2)
    T = T - T.mean(axis=0, keepdims=True) - T.mean(axis=2, keepdims=True) + T.mean(axis=(0, 2), keepdims=True)
    T = T.reshape(2 * n, 2 * n) / n
    T = T.at[jnp.diag_indices(2 * n)].add(diag_shift)
    if proj_reg is not None:
        T = T + proj_reg / (2 * n)

    e = local_energies.reshape(-1)
    dv = 2.0 * (e - e.mean()) / n**0.5
    dv = jnp.stack([dv.real, dv.imag], axis=-1).reshape(-1)
    if momentum is not None:
        dv = dv - momentum * centre(Ju)
    a = solver_fn(T, dv)
    a = centre(a[0] if isinstance(a, tuple) else a)

    # O^T a = J^T P a / sqrt(n_samples); the shift drops out because P a sums to zero
    if n_blocks == 1:
        def put(k, out):
            start = jnp.minimum(k * piece, n_params - piece)
            return jax.lax.dynamic_update_slice_in_dim(out, a @ jax.lax.dynamic_slice_in_dim(J, start, piece, 1),
                                                       start, 0)

        updates = jax.lax.fori_loop(0, -(-n_params // piece), put, jnp.zeros(n_params))
    else:
        def add_vjp(acc, xa):
            x, ak = xa
            return acc + jax.vjp(lambda th: re_im(th, x), theta)[1](ak)[0], None

        updates, _ = jax.lax.scan(add_vjp, jnp.zeros_like(theta), (chunks, a.reshape(*chunks.shape[:2], 2)))
    if momentum is not None:
        updates = updates + momentum * old_updates
    return unravel(updates), updates if momentum is not None else None


class BlockedVMC_SR(nk.driver.VMC_SR):
    """VMC_SR(use_ntk=True, mode="complex") with the update computed by blocked_srt.

    NetKet's dense NTK holds the Jacobian (2^13 samples x 2 x 116956 parameters = 14.3 GiB at
    triangular G=3) three times, and its on-the-fly NTK differentiates all samples at once,
    which for a C3v-projected state means 6 network evaluations each; both run out of GPU
    memory. blocked_srt holds one copy of one column block of at most block_bytes, and costs
    one Jacobian evaluation per block. Each chunk of samples is differentiated with respect to
    all parameters, so chunk_size_bwd is lowered until that is at most 2 GiB (64 samples at
    G=5, where the 1024 of the main stage would take 23 GiB). By default block_bytes is the
    free GPU memory, less two NTK-sized (2 n_samples)^2 matrices, one chunk and 1 GiB to
    spare: one block at G=3 on a 32 GB GPU, 12 at G=5. Without memory statistics (CPU), the
    Jacobian is one block.
    """

    def __init__(self, *args, block_bytes=None, **kwargs):
        super().__init__(*args, use_ntk=True, **kwargs)
        if self.mode != "complex" or nk.jax.tree_leaf_iscomplex(self.state.parameters):
            raise NotImplementedError("BlockedVMC_SR needs mode='complex' and real parameters")
        n, sample_bytes = self.state.n_samples, 2 * self.state.n_parameters * 8
        cap = min(self.chunk_size_bwd or n, max(1, 2**31 // sample_bytes))
        self.chunk_size_bwd = max(d for d in range(1, cap + 1) if n % d == 0)
        stats = jax.local_devices()[0].memory_stats() or {}
        if block_bytes is None and "bytes_limit" in stats:
            block_bytes = stats["bytes_limit"] - stats["bytes_in_use"] - 2 * 8 * (2 * n) ** 2 - 3 * 2**30
            if block_bytes <= 0:
                raise MemoryError(f"no GPU memory left for the Jacobian besides the NTK of {n} samples")
        self.n_blocks = 1 if block_bytes is None else -(-n * sample_bytes // block_bytes)

    def _forward_and_backward(self):
        """VMC_SR._forward_and_backward for an MCState, with blocked_srt."""
        self.state.reset()
        local_energies = self.state.local_estimators(self._ham)
        self._loss_stats = nk.stats.statistics(local_energies)
        diag_shift, proj_reg, momentum = (f(self.step_count) if callable(f) else f
                                          for f in (self.diag_shift, self.proj_reg, self.momentum))
        if momentum is not None and self._old_updates is None:
            self._old_updates = jnp.zeros(self.state.n_parameters)   # same jit trace as later steps
        self._dp, self._old_updates = blocked_srt(
            self.state._apply_fun, local_energies, self.state.parameters, self.state.model_state,
            self.state.samples.reshape(-1, self.state.hilbert.size), diag_shift=diag_shift,
            solver_fn=self._linear_solver, chunk_size=self.chunk_size_bwd, n_blocks=self.n_blocks,
            proj_reg=proj_reg, momentum=momentum, old_updates=self._old_updates)
        return self._dp


def lr_schedule(lr0, lr1, n_steps):
    """Learning rate at step k of a stage: constant lr0, or exponential decay from lr0 at
    the first step to lr1 at step n_steps - 1 (and lr1 after it)."""
    if lr1 is None or n_steps <= 1:
        return lambda k: lr0
    ratio = lr1 / lr0
    return lambda k: lr0 * ratio ** (jnp.minimum(k, n_steps - 1) / (n_steps - 1))


def run_stage(vstate, hi, H, n_steps, schedule, exact, args, desc, checkpoint=None):
    """n_steps of SR on vstate (exact sums, or Monte Carlo with BlockedVMC_SR), with learning
    rate schedule(i) at the i-th of these steps. Returns (energies, variances).

    checkpoint(vstate, energies, variances) is called every args.checkpoint_every steps.
    """
    energies, variances = [], []
    if n_steps <= 0:
        return energies, variances
    every = args.checkpoint_every
    if exact:
        step, flat, write_back = exact_sr_trainer(vstate, hi, H, args.diag_shift)
        with tqdm(total=n_steps, desc=desc, unit="it") as pbar:
            for i in range(n_steps):
                flat, E, var = step(flat, float(schedule(i)))
                energies.append(float(E))
                variances.append(float(var))
                pbar.set_postfix(E=f"{energies[-1]:.8f}", var=f"{variances[-1]:.2e}")
                pbar.update(1)
                if checkpoint is not None and every > 0 and len(energies) % every == 0:
                    write_back(flat)
                    checkpoint(vstate, energies, variances)
                if variances[-1] < args.var_tol:
                    tqdm.write(f"variance below {args.var_tol:g}, stopping")
                    break
        write_back(flat)
    else:
        optimizer = nk.optimizer.Sgd(learning_rate=schedule)
        vmc = BlockedVMC_SR(hamiltonian=H, optimizer=optimizer, diag_shift=args.diag_shift,
                            variational_state=vstate, momentum=0.8, mode="complex",
                            block_bytes=args.block_gib and int(args.block_gib * 2**30))
        desc += f" (Jacobian blocks: {vmc.n_blocks}, chunk: {vmc.chunk_size_bwd})"

        # The driver measures the energy on the samples it draws for each step, so
        # record that instead of calling vstate.expect(H) again (a second set of samples
        # and local energies per step). It is the energy before that step's update.
        def record(step, log_data, driver):
            st = log_data["Energy"]
            energies.append(float(np.real(st.mean)))
            variances.append(float(np.real(st.variance)))
            if checkpoint is not None and every > 0 and len(energies) % every == 0:
                checkpoint(vstate, energies, variances)
            return True

        print(desc)
        vmc.run(n_iter=n_steps, out=nk.logging.RuntimeLog(), show_progress=True, callback=record)
    return energies, variances


def thermalize(vstate, args, sweeps=None):
    """Sweep the chains before a stage (args.eval_discard times by default); training then
    continues from them with args.n_discard sweeps per step."""
    vstate.n_discard_per_chain = args.eval_discard if sweeps is None else sweeps
    vstate.sample()
    vstate.n_discard_per_chain = args.n_discard


def restore_chains(vstate, mpack):
    """Put the chains saved in mpack into vstate and return True, or return False if it holds
    none or a different number (vstate then keeps its chains from random configurations)."""
    with open(mpack, "rb") as f:
        saved = flax.serialization.msgpack_restore(f.read()).get("sampler_state")
    if saved is None or saved.get("σ") is None or saved["σ"].shape != vstate.sampler_state.σ.shape:
        print(f"WARNING: no matching chains in {mpack}, thermalising random ones instead")
        return False
    vstate.sampler_state = flax.serialization.from_state_dict(vstate.sampler_state, saved)
    vstate.reset()
    print(f"restored {saved['σ'].shape[0]} chains from {mpack}")
    return True


def drop_stuck_chains(vstate, margin, seed):
    """Replace chains with log|psi|^2 more than margin below the median by copies of other chains
    and return how many. Chains from random configurations can stay on a flat, low-amplitude
    plateau for hundreds of sweeps. At honeycomb G = 5 a single one, about 85 below the median with
    the next lowest chains at -30, made up nearly all of Var(H): 1-13 instead of 0.15-0.19."""
    σ = np.array(vstate.sampler_state.σ)
    log_p = 2 * np.real(np.asarray(vstate.log_value(σ)))
    stuck = log_p < np.median(log_p) - margin
    n = int(stuck.sum())
    if n:
        σ[stuck] = σ[np.random.default_rng(seed).choice(np.flatnonzero(~stuck), n)]
        vstate.sampler_state = vstate.sampler_state.replace(σ=jnp.asarray(σ, dtype=vstate.sampler_state.σ.dtype))
        vstate.reset()
    print(f"dropped {n} stuck chains (log|psi|^2 more than {margin:g} below the median)")
    return n


def write_checkpoint(path, vstate, info):
    """State and progress so far; written to temporary files first so an interrupted
    write never replaces a good checkpoint."""
    save_vstate(vstate, f"{path}.mpack.tmp")
    with open(f"{path}.json.tmp", "w") as f:
        json.dump(info, f)
    os.replace(f"{path}.mpack.tmp", f"{path}.mpack")
    os.replace(f"{path}.json.tmp", f"{path}.json")


def read_checkpoint(path):
    """Checkpoint JSON, also from earlier versions of this script."""
    ck = json.load(open(path))
    if "history" not in ck:   # first version: flat lists
        ck["history"] = {"main": {"energies": ck["energies"], "variances": ck["variances"]},
                         "c3v": {"energies": ck["symm_energies"], "variances": ck["symm_variances"]}}
        ck["milestones"] = {"before_c3v": ck["before_c3v"]} if ck.get("before_c3v") else {}
    if (ck["stage"] not in STAGES or ck.get("model", "plain") != "plain"
            or ck["history"].get("attn", {}).get("energies")):
        raise SystemExit(f"{path} comes from the removed symmetric-attention version and cannot be "
                         "resumed; use --restart")
    return ck


def load_network(vstate, mpack):
    """Network parameters of a saved state into vstate, whether either of them is C3v-projected
    or not (a projected state keeps the network's parameters under params/base_module)."""
    with open(mpack, "rb") as f:
        params = flax.serialization.msgpack_restore(f.read())["variables"]["params"]
    target = vstate.variables["params"]
    if "base_module" in params and "base_module" not in target:
        params = params["base_module"]
    elif "base_module" in target and "base_module" not in params:
        params = {"base_module": params}
    vstate.variables = {**vstate.variables, "params": flax.serialization.from_state_dict(target, params)}
    return vstate


def gpu_operator(H):
    """Jax version of a LocalOperator, so connected elements are computed on the GPU
    instead of with numba on the CPU. Falls back to H if NetKet cannot convert it."""
    try:
        return H.to_jax_operator()
    except (AttributeError, NotImplementedError, TypeError) as err:
        print(f"keeping the numba operator ({type(err).__name__}: {err})")
        return H


def evaluate(vstate, hi, H, exact, args):
    """(E, error, variance): exact sums, or MC on the current chains after args.eval_discard
    further sweeps. vstate.reset() only drops the stored samples: NetKet keeps the chains
    (reset_chains=False), so this is not an independent check from new chains."""
    if exact:
        E, var = exact_energy(exact_vector(vstate, hi), H.to_sparse())
        return E, 0.0, var
    vstate.n_discard_per_chain = args.eval_discard
    vstate.reset()
    st = vstate.expect(H)
    vstate.n_discard_per_chain = args.n_discard
    return float(st.mean.real), float(st.error_of_mean.real), float(st.variance.real)


def train(shape, G, seed, args):
    g, patches, x, y = build_fractal(shape, G, N_PATCH)
    distances = patch_distances(patches, x, y)
    N = g.n_nodes
    s0 = target_sz(N)
    # "always": projected throughout (6x cost per step); "final": plain ViT, then
    # c3v_steps projected steps at the end; "none": never projected
    c3v_main = args.c3v == "always"
    c3v_final = args.c3v == "final" and args.c3v_steps > 0
    c3v = c3v_main or c3v_final
    name = f"vit_{shape}_G={G}_seed={seed}"
    print(f"\n{'=' * 60}\n{name}: N={N}, S^z={s0:+}, C3v={args.c3v}\n{'=' * 60}")

    model_cfg = json.load(open(f"{_base(shape, G, seed)}_metadata.json"))["model"]
    out_dir = f"{args.out}/{shape}"
    os.makedirs(out_dir, exist_ok=True)
    ckpt = f"{out_dir}/{name}_ckpt"
    outfile = f"{out_dir}/{name}{'_symm' if c3v else ''}.mpack"
    # --continue: the finished state is renamed to <name>_prev* and trained further from there
    prev = outfile.replace(".mpack", "_prev")
    if args.continue_run and not os.path.exists(f"{ckpt}.json") and not args.restart:
        if os.path.exists(f"{prev}.mpack"):
            print(f"{outfile} was already continued from {prev}.mpack, skipping")
            return
        if not os.path.exists(outfile):
            print(f"--continue: {outfile} does not exist, skipping")
            return
        for suffix in (".mpack", "_metadata.json", "_energy_history.json"):
            if os.path.exists(outfile.replace(".mpack", suffix)):
                os.replace(outfile.replace(".mpack", suffix), prev + suffix)
        print(f"continuing the finished state, now saved as {prev}.mpack")
    if os.path.exists(outfile) and not os.path.exists(f"{ckpt}.json") and not args.restart:
        print(f"{outfile} exists, skipping (--restart to redo it, --continue to train it further)")
        return

    def build(stage):
        """Untrained state with the projection of the given stage."""
        symm = c3v_main or stage == "c3v"
        vs, hi = sector_mcstate(g, patches, distances, model_cfg, s0, False, args.n_chains,
                                args.n_samples, args.n_discard, args.chunk_size, seed=seed)
        return (project_c3v(vs, hi, g, args.c3v_chunk_size, seed=seed) if symm else vs), hi

    chains_restored = False
    ck = None
    if os.path.exists(f"{ckpt}.json") and not args.restart:
        ck = read_checkpoint(f"{ckpt}.json")
        if ck["c3v_mode"] != args.c3v:
            raise SystemExit(f"{ckpt}.json was written with --c3v {ck['c3v_mode']}; use the same, "
                             "or --restart to discard it")

    if ck is not None:
        stage = ck["stage"]
        vstate, hi = build(stage)
        load_variables(vstate, f"{ckpt}.mpack")
        chains_restored = restore_chains(vstate, f"{ckpt}.mpack")
        warm, hist, milestones, elapsed_before = ck["warm"], ck["history"], ck["milestones"], ck["elapsed"]
        print(f"resuming from {ckpt}.mpack in stage {stage}; steps done: "
              + ", ".join(f"{s} {len(hist[s]['energies'])}" for s in STAGES))
    else:
        stage = "main"
        hist = {s: {"energies": [], "variances": []} for s in STAGES}
        milestones, elapsed_before = {}, 0.0
        if args.from_scratch:
            vstate, hi = build("main")
            warm = {"source": None}
        elif args.continue_run:
            vstate, hi = build("main")
            load_network(vstate, f"{prev}.mpack")
            chains_restored = restore_chains(vstate, f"{prev}.mpack")
            warm = {"source": f"{prev}.mpack", "continued": True, "flipped": False, "weight_kept": 1.0,
                    "chains": "restored" if chains_restored else "fresh"}
            print(f"warm start from {prev}.mpack")
        else:
            # c3v_main warm-starts from the saved projected state, otherwise from the plain one
            vstate, hi, warm = load_sector_state(shape, G, seed, g, patches, distances, c3v_main, args.n_chains,
                                                 args.n_samples, args.eval_discard, args.chunk_size)
            vstate.n_discard_per_chain = args.n_discard
            print(f"warm start from {warm['source']}, kept {warm['weight_kept']:.3f} of it"
                  f"{', spin-flipped' if warm['flipped'] else ''}")

    if chains_restored and args.drop_stuck_chains is not None:
        n_dropped = drop_stuck_chains(vstate, args.drop_stuck_chains, seed)
        warm = {**warm, "chains_dropped": warm.get("chains_dropped", 0) + n_dropped}

    H = build_hamiltonian(hi, g)
    reference = reference_energy(shape, G, g, s0)
    e_ed = reference[0] if reference else None
    if e_ed is not None:
        print(f"exact sector ground state: {e_ed:.10f}")

    dim = comb(N, int(round(N / 2 + s0)))
    exact = dim <= FULLSUM_MAX and not args.mc
    print(f"sector has {dim} states -> {'exact sums' if exact else 'Monte Carlo'}")
    lr = args.lr if args.lr is not None else (1e-2 if exact else 1e-3)
    lr_final = args.lr_final
    c3v_lr = args.c3v_lr if args.c3v_lr is not None else (lr_final if lr_final is not None else lr)
    main_schedule = lr_schedule(lr, lr_final, args.n_opt)
    print(f"learning rate: main stage {lr:g}"
          + (f" -> {lr_final:g} (exponential decay)" if lr_final is not None else " (constant)")
          + (f", C3v stage {c3v_lr:g}" if c3v_final else ""))
    # exact mode needs H as a matrix; Monte Carlo runs faster with the jax operator
    H_run = H if exact or args.numba_operator else gpu_operator(H)

    start = time.time()

    def checkpoint_for(stage_name):
        def save(vs, e, v):
            h = {s: {"energies": list(d["energies"]), "variances": list(d["variances"])} for s, d in hist.items()}
            h[stage_name]["energies"] += e
            h[stage_name]["variances"] += v
            write_checkpoint(ckpt, vs, {"stage": stage_name, "c3v_mode": args.c3v, "warm": warm, "history": h,
                                        "milestones": milestones,
                                        "elapsed": elapsed_before + time.time() - start})
        return save

    def train_stage(vs, stage_name, n_total, schedule):
        """Run stage_name up to n_total steps in all (fewer if resumed); schedule(k) is the
        learning rate at step k of the stage, counted from its first step."""
        nonlocal chains_restored
        sweeps = RESTORED_DISCARD if chains_restored else None   # only the first stage after restoring
        chains_restored = False
        done = len(hist[stage_name]["energies"])
        if done >= n_total:
            # finished before an interruption (e.g. during the evaluation): no thermalisation needed
            print(f"{stage_name} stage already complete ({done} steps)")
            return
        if not exact:
            thermalize(vs, args, sweeps)
        save = checkpoint_for(stage_name)
        e, v = run_stage(vs, vs.hilbert, H_run, n_total - done, lambda i: schedule(done + i), exact, args,
                         f"{name} [{stage_name}]", save)
        # Always keep the state after the last step, whatever --checkpoint-every is, so a job
        # killed later (in the evaluation that follows) resumes without repeating any step.
        if e and (args.checkpoint_every <= 0 or len(e) % args.checkpoint_every != 0):
            save(vs, e, v)
        hist[stage_name]["energies"] += e
        hist[stage_name]["variances"] += v

    def milestone(vs, key):
        E, err, var = evaluate(vs, vs.hilbert, H_run, exact, args)
        milestones[key] = {"energy": E, "error": err, "variance": var}
        print(f"{key}: E = {E:.8f} +- {err:.1e}, var = {var:.2e}")

    if stage == "main":
        train_stage(vstate, "main", args.n_opt, main_schedule)
        if c3v_final:
            milestone(vstate, "before_c3v")
            vstate = project_c3v(vstate, vstate.hilbert, g, args.c3v_chunk_size, seed=seed)
            stage = "c3v"
            checkpoint_for("c3v")(vstate, [], [])

    if stage == "c3v":
        train_stage(vstate, "c3v", args.c3v_steps, lambda k: c3v_lr)
    elapsed = elapsed_before + time.time() - start

    E_final, err_final, var_final = evaluate(vstate, vstate.hilbert, H_run, exact, args)
    print(f"final E = {E_final:.8f} +- {err_final:.1e}, var = {var_final:.2e}"
          + (f", exact {e_ed:.8f} (diff {E_final - e_ed:.1e})" if e_ed is not None else ""))

    save_vstate(vstate, outfile)
    (main_e, main_v), (c3v_e, c3v_v) = ((hist[s]["energies"], hist[s]["variances"]) for s in STAGES)
    all_energies = main_e + c3v_e
    save_metadata(
        outfile.replace(".mpack", "_metadata.json"),
        model_cfg,
        {"type": "MetropolisExchange", "n_chains": args.n_chains, "n_samples": args.n_samples,
         "n_discard_per_chain": args.n_discard, "eval_n_discard_per_chain": args.eval_discard,
         "seed": seed, "total_sz": s0, "c3v": c3v, "chunk_size": args.chunk_size},
        {"J": 1.0, "Jz": 0.0, "hx": 0.0, "hy": 0.0},
        optimizer_config={"type": "exact SR (full sum)" if exact else "Sgd+VMC_SR", "lr": lr,
                          "lr_final": lr_final, "lr_schedule": "exponential" if lr_final is not None else "constant",
                          "diag_shift": args.diag_shift, "momentum": None if exact else 0.8,
                          "n_opt": args.n_opt, "steps_done": len(main_e),
                          "c3v_mode": args.c3v, "c3v_lr": c3v_lr if c3v_final else None,
                          "c3v_steps_done": len(c3v_e)},
        final_results={
            "exact": bool(exact),
            "final_energy": E_final,
            "final_energy_error": err_final,
            "final_variance": var_final,
            "train_energy_last50": float(np.mean(all_energies[-50:])) if all_energies else None,
            "ed_sector_energy": None if e_ed is None else float(e_ed),
            "before_c3v": milestones.get("before_c3v"),
            "milestones": milestones,
            "training_time": elapsed,
        },
        fractal={"shape": shape, "G": G, "n": N_PATCH},
        warm_start=warm,
    )
    with open(outfile.replace(".mpack", "_energy_history.json"), "w") as f:
        json.dump({"energies": main_e, "variances": main_v, "symm_start_iter": len(main_e),
                   "symm_energies": c3v_e, "symm_variances": c3v_v}, f, indent=2)
    print(f"saved {outfile}")
    for ext in (".mpack", ".json"):
        if os.path.exists(ckpt + ext):
            os.remove(ckpt + ext)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shapes", nargs="+", default=["triangular", "honeycomb"])
    p.add_argument("--gens", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--c3v", choices=["final", "always", "none"], default="final",
                   help="final: plain ViT, then --c3v-steps projected steps (fast); "
                        "always: projected throughout (6x slower); none: never projected")
    p.add_argument("--c3v-steps", type=int, default=50, help="projected steps at the end for --c3v final")
    p.add_argument("--from-scratch", action="store_true", help="random initialisation instead of warm start")
    p.add_argument("--n-opt", type=int, default=300, help="steps of the main stage")
    p.add_argument("--lr", type=float, default=None,
                   help="(initial) learning rate of the main stage; default 1e-2 exact, 1e-3 Monte Carlo")
    p.add_argument("--lr-final", type=float, default=None,
                   help="learning rate at the last main-stage step, reached by exponential decay from --lr "
                        "(default: constant --lr)")
    p.add_argument("--c3v-lr", type=float, default=None,
                   help="learning rate of the C3v stage (default: --lr-final, else --lr)")
    p.add_argument("--mc", action="store_true", help=f"sample even sectors with <= {FULLSUM_MAX} states")
    p.add_argument("--var-tol", type=float, default=1e-10, help="exact mode: stop once Var(H) is below this")
    p.add_argument("--diag-shift", type=float, default=1e-4)
    p.add_argument("--n-chains", type=int, default=2**13)
    p.add_argument("--n-samples", type=int, default=2**13)
    p.add_argument("--n-discard", type=int, default=0, help="sweeps per chain between training steps "
                   "(chains persist, so 0 is fine, as in trainViT.py)")
    p.add_argument("--eval-discard", type=int, default=64, help="thermalisation sweeps for evaluation")
    p.add_argument("--chunk-size", type=int, default=1024,
                   help="samples per batch on the GPU; lower it if you run out of memory (trainViT.py used 128)")
    p.add_argument("--c3v-chunk-size", type=int, default=None,
                   help="chunk of the projected state, which evaluates all 6 C3v images of each sample "
                        "(default: --chunk-size // 6; rounded down to a divisor of --n-samples)")
    p.add_argument("--block-gib", type=float, default=None,
                   help="GPU memory for one column block of the Jacobian in the Monte Carlo SR step "
                        "(default: all that is free besides the NTK; each block costs one Jacobian "
                        "evaluation, see BlockedVMC_SR)")
    p.add_argument("--numba-operator", action="store_true",
                   help="keep the CPU (numba) Hamiltonian instead of converting it to a jax operator")
    p.add_argument("--checkpoint-every", type=int, default=25,
                   help="save state and history every this many steps (0: never); rerunning resumes from it")
    p.add_argument("--restart", action="store_true",
                   help="start over, ignoring an existing checkpoint or finished state")
    p.add_argument("--continue", dest="continue_run", action="store_true",
                   help="train a finished run further: its state is renamed to <name>_prev* and is the "
                        "warm start of a new main stage (--n-opt more steps) and C3v stage")
    p.add_argument("--drop-stuck-chains", type=float, default=None, metavar="MARGIN",
                   help="after restoring saved chains (resume or --continue), replace those with "
                        "log|psi|^2 more than MARGIN below the median by copies of other chains")
    p.add_argument("--out", default="data/ViT_sz")
    args = p.parse_args()

    print(f"NetKet {nk.__version__}, Jax {jax.__version__}, devices {jax.devices()}")
    for shape in args.shapes:
        for G in args.gens:
            for seed in args.seeds:
                train(shape, G, seed, args)
