"""Option 2: compare seeds inside a single S^z sector.

The XY Hamiltonian conserves S^z, so the ground state of each sector is a
ground-state candidate on its own. Evaluating a trained network only on
configurations with S^z = s0 (a constrained Hilbert space) gives its projection
P_s0 |psi>. Sectors with at most FULLSUM_MAX states are summed exactly (energies and
fidelities without sampling error); larger ones are sampled.

  even N: s0 = 0
  odd N:  s0 = +1/2. Seeds that mostly live in S^z = -1/2 are first mapped
          there by the spin flip psi(s) -> psi(-s), a symmetry of H, so that
          every seed is compared in the same sector.

For every (shape, G) this writes <source>/<shape>/fidelity_<shape>_gasket_G=<G>_sector[_symm].json
with the pairwise F (both MC directions), the sector energy of each seed, and how much of
each original state was kept by the projection. For N <= ED_MAX_N the exact sector energy
is stored as a reference.

    python sector_fidelity.py                      # C3v-projected states (raw ones at G=5)
    python sector_fidelity.py --raw                # unprojected states
    python sector_fidelity.py --source data/ViT_sz # states retrained by trainViT_sz.py
    python sector_fidelity.py --shapes honeycomb --gens 1 2
"""

import argparse
import gc
import json
import os
from math import comb

import flax
import jax
import netket as nk
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from netket.experimental.observable import InfidelityOperator

from expect import load_vstate_vit, load_vstate_vit_symm
from trainViT import ViT, build_fractal, patch_distances
from utils import build_hamiltonian

N_PATCH = 2
ED_MAX_N = 18
FULLSUM_MAX = 10000  # sectors up to this many states are summed exactly instead of sampled
RESTORED_DISCARD = 16  # sweeps per chain when starting from the saved, equilibrated chains


def _base(shape, G, seed, root="data/ViT"):
    return f"{root}/{shape}/vit_{shape}_G={G}_seed={seed}"


def target_sz(N):
    return 0.0 if N % 2 == 0 else 0.5


def flip_spins(variables):
    """Variables of psi'(s) = psi(-s).

    The spins enter the ViT only through the Embed Dense layer, W x + b, so
    negating its kernel is exactly the global spin flip. Works for projected
    states too, because site permutations commute with the flip.
    """
    hits = []

    def negate(path, x):
        names = [str(getattr(k, "key", k)) for k in path]
        if names[-1] == "kernel" and any(n.startswith("Embed") for n in names):
            hits.append("/".join(names))
            return -x
        return x

    out = jax.tree_util.tree_map_with_path(negate, variables)
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one Embed kernel, found {hits}")
    return out


def load_variables(vstate, mpack):
    """Copy only the network variables from a saved MCState. For the data/ViT states the saved
    sampler state holds chains from the unconstrained space and must not be reused
    (load_retrained_state restores the in-sector chains of trainViT_sz.py separately)."""
    with open(mpack, "rb") as f:
        raw = flax.serialization.msgpack_restore(f.read())
    vstate.variables = flax.serialization.from_state_dict(vstate.variables, raw["variables"])
    return vstate


def sector_mcstate(g, patches, distances, model_cfg, total_sz, symm, n_chains, n_samples,
                   n_discard, chunk_size=128, seed=0):
    """Untrained ViT on the S^z = total_sz sector, sampled with nearest-neighbour exchanges."""
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes, total_sz=total_sz)
    model = ViT(
        num_layers=model_cfg["num_layers"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        patches=tuple(map(tuple, patches)),
        distances=tuple(map(tuple, distances)),
    )
    sampler = nk.sampler.MetropolisExchange(hi, graph=g, n_chains=n_chains)
    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard,
                            chunk_size=chunk_size, seed=seed, sampler_seed=seed)
    if symm:
        vstate = project_c3v(vstate, hi, g, seed=seed)
    return vstate, hi


def project_c3v(vstate, hi, g, chunk_size=None, seed=0):
    """Project vstate onto the trivial irrep of the lattice automorphisms, with its sampler
    seeded by seed (rep.project would seed it at random, so runs would not be reproducible).
    The projected state evaluates the network on every group image of each sample, so by
    default its chunk is vstate.chunk_size // |group| to keep the same memory per batch
    (rounded down to a divisor of n_samples)."""
    group = g.automorphisms()
    rep = nk.symmetry.canonical_representation(hilbert=hi, group=group)
    proj = nk.vqs.apply_operator(rep.projector(0), vstate, seed=seed)
    if chunk_size is None and vstate.chunk_size is not None:
        chunk_size = max(1, vstate.chunk_size // len(group))
    if chunk_size is not None:
        # NetKet's SR solver needs a chunk that divides n_samples: take the largest such one
        chunk_size = max(d for d in range(1, chunk_size + 1) if vstate.n_samples % d == 0)
    proj.chunk_size = chunk_size
    proj.n_discard_per_chain = vstate.n_discard_per_chain
    return proj


def sector_weights(vstate, N):
    """Fraction of MC samples in each S^z sector (NetKet spins are +-1)."""
    s = np.asarray(vstate.sample()).reshape(-1, N)
    sz, counts = np.unique(s.sum(axis=1) / 2, return_counts=True)
    return {float(k): float(c) / len(s) for k, c in zip(sz, counts)}


def load_sector_state(shape, G, seed, g, patches, distances, symm=True, n_chains=2**10,
                      n_samples=2**13, n_discard=256, chunk_size=128):
    """Trained state of one seed, restricted to S^z = target_sz(N).

    symm=True uses the C3v-projected state; if none was saved (G=5) the raw
    state is loaded and projected here. Returns (vstate, hilbert, info).
    """
    N = g.n_nodes
    s0 = target_sz(N)
    base = _base(shape, G, seed)
    meta_file = f"{base}_metadata.json"
    model_cfg = json.load(open(meta_file))["model"]
    have_symm = os.path.exists(f"{base}_symm.mpack")
    use_symm_file = symm and have_symm

    # sector content of the original state, to pick +1/2 or -1/2 and report the kept weight
    hi_full = nk.hilbert.Spin(s=0.5, N=N)
    if use_symm_file:
        full = load_vstate_vit_symm(f"{base}_symm.mpack", meta_file, hi_full, g, patches, distances,
                                    chunk_size=chunk_size)
    else:
        full = load_vstate_vit(f"{base}.mpack", meta_file, hi_full, patches, distances,
                               chunk_size=chunk_size)
    full.n_samples = n_samples
    full.n_discard_per_chain = n_discard
    weights = sector_weights(full, N)
    del full

    flipped = s0 > 0 and weights.get(-s0, 0.0) > weights.get(s0, 0.0)

    vstate, hi = sector_mcstate(g, patches, distances, model_cfg, s0, use_symm_file, n_chains,
                                n_samples, n_discard, chunk_size, seed)
    load_variables(vstate, f"{base}_symm.mpack" if use_symm_file else f"{base}.mpack")
    if symm and not use_symm_file:
        vstate = project_c3v(vstate, hi, g, seed=seed)
    if flipped:
        vstate.variables = flip_spins(vstate.variables)
    vstate.reset()

    info = {
        "source": f"{base}_symm.mpack" if use_symm_file else f"{base}.mpack",
        "c3v": bool(symm),
        "flipped": bool(flipped),
        "sector_weights": weights,
        "weight_kept": weights.get(-s0 if flipped else s0, 0.0),
    }
    return vstate, hi, info


def load_retrained_state(shape, G, seed, g, patches, distances, root, n_chains=2**10,
                         n_samples=2**13, n_discard=256, chunk_size=128, restored_discard=RESTORED_DISCARD):
    """State saved by trainViT_sz.py. It already lives in its S^z sector, so no
    sector weights or spin flip; total_sz and C3v are read from its metadata.

    The saved sampler state holds the chains of the final evaluation, already in the sector
    and in equilibrium, so they are restored (n_chains is then taken from them) and only
    restored_discard sweeps are discarded. Fresh chains from random configurations need far
    more sweeps at G >= 4 (energies off by up to 20 at G = 5 with 64), hence n_discard = 256
    for them, as in the final evaluation of trainViT_sz.py."""
    base = _base(shape, G, seed, root)
    mpack = f"{base}_symm.mpack" if os.path.exists(f"{base}_symm.mpack") else f"{base}.mpack"
    meta = json.load(open(mpack.replace(".mpack", "_metadata.json")))
    if meta["model"].get("attention", "plain") != "plain":
        raise ValueError(f"{mpack} was trained with the removed symmetric-attention ViT")
    with open(mpack, "rb") as f:
        saved = flax.serialization.msgpack_restore(f.read()).get("sampler_state")
    restored = saved is not None and saved.get("σ") is not None
    if restored:
        n_chains, n_discard = saved["σ"].shape[0], restored_discard
    else:
        print(f"  {mpack}: no saved chains, thermalising fresh ones with {n_discard} sweeps; "
              "check that the energy matches the training")
    sampler_cfg = meta["sampler"]
    vstate, hi = sector_mcstate(g, patches, distances, meta["model"], sampler_cfg["total_sz"],
                                sampler_cfg["c3v"], n_chains, n_samples, n_discard, chunk_size, seed)
    load_variables(vstate, mpack)
    if restored:
        vstate.sampler_state = flax.serialization.from_state_dict(vstate.sampler_state, saved)
    vstate.reset()
    info = {"source": mpack, "c3v": bool(sampler_cfg["c3v"]), "flipped": False,
            "sector_weights": None, "weight_kept": 1.0,
            "chains": "restored" if restored else "fresh", "n_chains": int(n_chains),
            "n_discard_per_chain": int(n_discard)}
    return vstate, hi, info


def ed_sector_energy(g, total_sz, k=2):
    """Lowest k eigenvalues of H (J=1, Jz=0) in the S^z = total_sz sector."""
    N = g.n_nodes
    n_up = int(round(N / 2 + total_sz))
    states = np.array([s for s in range(2**N) if bin(s).count("1") == n_up])
    pos = {s: i for i, s in enumerate(states)}
    rows, cols = [], []
    for i, s in enumerate(states):
        for a, b in g.edges():
            if ((s >> a) & 1) != ((s >> b) & 1):
                rows.append(i)
                cols.append(pos[s ^ ((1 << a) | (1 << b))])
    # -J (sx sx + sy sy) = -2J (s+ s- + s- s+) with Pauli matrices
    H = sp.csr_matrix((-2.0 * np.ones(len(rows)), (rows, cols)), shape=(len(states),) * 2)
    if len(states) < 500:
        return np.linalg.eigvalsh(H.toarray())[:k]
    return np.sort(sla.eigsh(H, k=k, which="SA")[0])


def exact_vector(vstate, hi):
    """Normalised amplitudes of vstate on every configuration of hi (small sectors only)."""
    lp = np.asarray(vstate.log_value(hi.all_states()))
    psi = np.exp(lp - lp.real.max())
    return psi / np.linalg.norm(psi)


def exact_energy(psi, H_sparse):
    """(E, variance) of a normalised vector."""
    Hpsi = H_sparse @ psi
    E = float(np.vdot(psi, Hpsi).real)
    return E, float(np.vdot(Hpsi, Hpsi).real - E**2)


def reference_energy(shape, G, g, total_sz):
    """Exact lowest energies of the sector: data/ED/ (ed_sector.py) if present, else ED here
    for N <= ED_MAX_N, else None."""
    path = f"data/ED/{shape}_G={G}_sz={total_sz:+g}.json"
    if os.path.exists(path):
        return json.load(open(path))["energies"]
    if g.n_nodes <= ED_MAX_N:
        return ed_sector_energy(g, total_sz).tolist()
    return None


def fidelity(a, b, n_samples):
    """F(a, b) estimated by sampling b. Returns (F, error)."""
    b.n_samples = n_samples
    st = b.expect(InfidelityOperator(a, cv_coeff=None))
    return 1.0 - complex(st.mean).real, float(np.real(st.error_of_mean))


def run(shape, G, seeds, symm, args):
    g, patches, x, y = build_fractal(shape, G, N_PATCH)
    distances = patch_distances(patches, x, y)
    N = g.n_nodes
    s0 = target_sz(N)
    dim = comb(N, int(round(N / 2 + s0)))
    exact = dim <= FULLSUM_MAX and not args.mc
    print(f"\n{shape} G={G}: N={N}, sector S^z={s0:+}, {dim} states"
          f" -> {'exact sums' if exact else 'Monte Carlo'}")

    states, infos, energies = [], {}, {}
    H = H_sparse = None
    for seed in seeds:
        if args.source == "data/ViT":
            vs, hi, info = load_sector_state(shape, G, seed, g, patches, distances, symm, args.n_chains,
                                             args.n_samples, args.n_discard)
        else:
            vs, hi, info = load_retrained_state(shape, G, seed, g, patches, distances, args.source,
                                                args.n_chains, args.n_samples, args.n_discard)
            symm = info["c3v"]
        if H is None:
            H = build_hamiltonian(hi, g)
            H_sparse = H.to_sparse() if exact else None
        if exact:
            vs = exact_vector(vs, hi)
            E, var = exact_energy(vs, H_sparse)
            energies[seed] = {"mean": E, "error": 0.0, "variance": var}
        else:
            st = vs.expect(H)
            energies[seed] = {"mean": float(st.mean.real), "error": float(st.error_of_mean.real),
                              "variance": float(st.variance.real)}
        infos[seed] = info
        states.append(vs)
        print(f"  seed {seed}: E = {energies[seed]['mean']:.6f} +- {energies[seed]['error']:.1e}, "
              f"var = {energies[seed]['variance']:.1e}, "
              f"kept {info['weight_kept']:.3f} of the state{'  (flipped)' if info['flipped'] else ''}")

    e_ed = reference_energy(shape, G, g, s0)
    if e_ed is not None:
        print(f"  exact sector energies: {e_ed}")
        for seed, e in energies.items():
            sigma = f"{(e['mean'] - e_ed[0]) / e['error']:+.1f} sigma" if e["error"] > 0 else "exact"
            print(f"  seed {seed}: E - E_exact = {e['mean'] - e_ed[0]:+.2e} ({sigma})")

    fid, fid_ij, fid_ji, err = {}, {}, {}, {}
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            key = f"seed{seeds[i]}_seed{seeds[j]}"
            if exact:
                fid_ij[key] = fid_ji[key] = float(abs(np.vdot(states[i], states[j])) ** 2)
                e1 = e2 = 0.0
            else:
                fid_ij[key], e1 = fidelity(states[i], states[j], args.n_samples)
                fid_ji[key], e2 = fidelity(states[j], states[i], args.n_samples)
            fid[key] = 0.5 * (fid_ij[key] + fid_ji[key])
            err[key] = 0.5 * float(np.hypot(e1, e2))
            print(f"  F({seeds[i]},{seeds[j]}) = {fid_ij[key]:.5f} / {fid_ji[key]:.5f}")

    values = list(fid.values()) or [np.nan]   # a single seed has no pairs
    results = {
        "shape": shape,
        "generation": G,
        "n_sites": N,
        "sector_sz": s0,
        "c3v": bool(symm),
        "seeds": seeds,
        "fidelities": fid,           # same layout as fidelity.py, so pp.py / degeneracy.py can read it
        "fidelities_ij": fid_ij,
        "fidelities_ji": fid_ji,
        "fidelity_errors": err,
        "mean_fidelity": float(np.nanmean(values)) if fid else None,
        "std_fidelity": float(np.nanstd(values)) if fid else None,
        "min_fidelity": float(np.nanmin(values)) if fid else None,
        "max_fidelity": float(np.nanmax(values)) if fid else None,
        "energies": energies,
        "ed_sector_energies": e_ed,
        "states": infos,
        "exact": bool(exact),
        "sampling": {"n_chains": args.n_chains, "n_samples": args.n_samples, "n_discard_per_chain": args.n_discard},
    }
    results["source"] = args.source
    out = f"{args.source}/{shape}/fidelity_{shape}_gasket_G={G}_sector{'_symm' if symm else ''}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  mean F = {results['mean_fidelity']:.5f} +- {results['std_fidelity']:.5f}  -> {out}" if fid
          else f"  one seed, no fidelities -> {out}")

    del states
    gc.collect()
    jax.clear_caches()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shapes", nargs="+", default=["triangular", "honeycomb"])
    p.add_argument("--gens", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--raw", action="store_true", help="use the unprojected states instead of the C3v ones")
    p.add_argument("--n-chains", type=int, default=2**10)
    p.add_argument("--n-samples", type=int, default=2**13)
    p.add_argument("--n-discard", type=int, default=256,
                   help="thermalisation sweeps per chain for fresh chains (saved chains of retrained "
                        f"states are restored and need only {RESTORED_DISCARD}); 64 is too few at G >= 4")
    p.add_argument("--mc", action="store_true", help=f"sample even sectors with <= {FULLSUM_MAX} states")
    p.add_argument("--source", default="data/ViT",
                   help="data/ViT: original states, projected here; data/ViT_sz: states retrained by trainViT_sz.py")
    args = p.parse_args()

    print(f"NetKet {nk.__version__}, Jax {jax.__version__}, devices {jax.devices()}")
    for shape in args.shapes:
        for G in args.gens:
            run(shape, G, args.seeds, not args.raw, args)
