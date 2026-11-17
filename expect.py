import json
from functools import partial

import numpy as np
import flax
import netket as nk
from flax import linen as nn
from netket.operator.spin import sigmax, sigmay, sigmaz

from utils import (build_hamiltonian, get_distance_shells, get_euclidean_distance_shells_all,
                   build_corner_path_correlators, get_corner_path_pairs)


def xy_correlator(hi, pairs):
    return sum(sigmax(hi, i) @ sigmax(hi, j) + sigmay(hi, i) @ sigmay(hi, j) for i, j in pairs) / len(pairs)


def build_observables(hi, g, positions=None, H=None, transverse=True):
    """transverse=False leaves out Mx and My (total and per site), which change S^z: they
    vanish in a fixed-S^z state and cannot be evaluated on a constrained Hilbert space."""
    N = g.n_nodes
    if H is None:
        H = build_hamiltonian(hi, g)

    obs_dict = {"E": H}
    if transverse:
        obs_dict["Mx"] = sum(sigmax(hi, i) for i in range(N)) / N
        obs_dict["My"] = sum(sigmay(hi, i) for i in range(N)) / N
    obs_dict["Mz"] = sum(sigmaz(hi, i) for i in range(N)) / N

    for d, pairs in get_distance_shells(g).items():
        print(f"Graph distance shell d={d}: {len(pairs)} pairs")
        obs_dict[f"Cxy_graph_r{d}"] = xy_correlator(hi, pairs)

    if positions is not None:
        obs_dict.update(build_corner_path_correlators(hi, g, positions))
        for d, pairs in sorted(get_euclidean_distance_shells_all(positions).items()):
            print(f"Euclidean distance shell d={d:.4f}: {len(pairs)} pairs")
            obs_dict[f"Cxy_euclidean_r{d:.4f}"] = xy_correlator(hi, pairs)

    for i in range(N):
        if transverse:
            obs_dict[f"Mx_{i}"] = sigmax(hi, i)
            obs_dict[f"My_{i}"] = sigmay(hi, i)
        obs_dict[f"Mz_{i}"] = sigmaz(hi, i)

    return obs_dict


def correlator_pair_groups(g, positions=None):
    """The XY correlators of build_observables as {name: pairs}, same names and order. A pair may
    appear several times in one group (corner paths), and then counts that many times."""
    groups = {}
    for d, pairs in get_distance_shells(g).items():
        print(f"Graph distance shell d={d}: {len(pairs)} pairs")
        groups[f"Cxy_graph_r{d}"] = pairs
    if positions is not None:
        for d, pairs in get_corner_path_pairs(g, positions).items():
            groups[f"Cxy_corner_d{d}"] = pairs
        for d, pairs in sorted(get_euclidean_distance_shells_all(positions).items()):
            print(f"Euclidean distance shell d={d:.4f}: {len(pairs)} pairs")
            groups[f"Cxy_euclidean_r{d:.4f}"] = pairs
    return groups


def xy_correlators(vstate, groups, batch=1024, log_every=600):
    """<C_xy> averaged over each group of pairs, from the current samples of vstate, with
    C_xy(i, j) = sx_i sx_j + sy_i sy_j. Same estimator and statistics as vstate.expect on the
    summed operator, but each distinct pair is evaluated once per sample, however many groups it
    is in, and only where it is not zero: its local value is 2 psi(s^ij) / psi(s) if s_i != s_j
    (s^ij: s with the two spins swapped), else 0. batch is the number of swapped configurations
    per network call (times |group| network evaluations for a C3v-projected state)."""
    import time

    import jax
    import jax.numpy as jnp

    # distinct unordered pairs, and for each the groups it is in, with multiplicity
    names = list(groups)
    pair_id, members = {}, []
    for k, pairs in enumerate(names):
        for i, j in groups[pairs]:
            key = (min(i, j), max(i, j))
            p = pair_id.setdefault(key, len(pair_id))
            if p == len(members):
                members.append({})
            members[p][k] = members[p].get(k, 0) + 1
    n_pairs, n_groups = len(pair_id), len(names)
    width = max(len(m) for m in members)
    memb_k = np.zeros((n_pairs, width), dtype=np.int32)
    memb_w = np.zeros((n_pairs, width))
    for p, m in enumerate(members):
        for c, (k, w) in enumerate(m.items()):
            memb_k[p, c], memb_w[p, c] = k, w
    counts = np.array([len(groups[n]) for n in names], dtype=float)
    I, J = (np.array(a, dtype=np.int32) for a in zip(*pair_id))

    σ = vstate.samples
    shape = σ.shape[:-1]
    σ = np.asarray(σ).reshape(-1, σ.shape[-1])
    n_samples = len(σ)
    apply_fun, variables = vstate._apply_fun, vstate.variables

    log_psi = jax.jit(apply_fun)
    logψ0 = jnp.concatenate([log_psi(variables, jnp.asarray(σ[a:a + batch])) for a in range(0, n_samples, batch)])

    @partial(jax.jit, donate_argnums=(1,))
    def step(variables, acc, σ, logψ0, I, J, memb_k, memb_w, sidx, p, valid):
        s = σ[sidx]
        i, j = I[p], J[p]
        rows = jnp.arange(len(sidx))
        x = s.at[rows, i].set(s[rows, j]).at[rows, j].set(s[rows, i])
        val = jnp.where(valid, 2 * jnp.exp(apply_fun(variables, x) - logψ0[sidx]), 0)
        for c in range(memb_k.shape[1]):
            acc = acc.at[sidx, memb_k[p, c]].add(val * memb_w[p, c])
        return acc

    dev = [jnp.asarray(a) for a in (σ, I, J, memb_k, memb_w)]
    σ_d, I_d, J_d, mk_d, mw_d = dev
    acc = jnp.zeros((n_samples, n_groups), dtype=jnp.result_type(logψ0.dtype, jnp.complex64))

    # nonzero (sample, pair) entries, streamed in fixed-size batches
    rows_per_pass = max(1, 2**24 // n_pairs)
    total = sum(int((σ[a:a + rows_per_pass, I] != σ[a:a + rows_per_pass, J]).sum())
                for a in range(0, n_samples, rows_per_pass))
    print(f"  correlators: {n_pairs} distinct pairs in {n_groups} groups, {n_samples} samples, "
          f"{total} nonzero pair terms to evaluate in batches of {batch}", flush=True)

    def flush(s, p):
        nonlocal acc
        valid = np.zeros(batch, dtype=bool)
        valid[:len(s)] = True
        s = np.pad(s, (0, batch - len(s)))
        p = np.pad(p, (0, batch - len(p)))
        acc = step(variables, acc, σ_d, logψ0, I_d, J_d, mk_d, mw_d, jnp.asarray(s), jnp.asarray(p),
                   jnp.asarray(valid))

    pending_s = np.zeros(0, dtype=np.int32)
    pending_p = np.zeros(0, dtype=np.int32)
    done, t0, last = 0, time.time(), time.time()
    for a in range(0, n_samples, rows_per_pass):
        s_idx, p_idx = np.nonzero(σ[a:a + rows_per_pass, I] != σ[a:a + rows_per_pass, J])
        pending_s = np.concatenate([pending_s, (s_idx + a).astype(np.int32)])
        pending_p = np.concatenate([pending_p, p_idx.astype(np.int32)])
        n_full = len(pending_s) // batch * batch
        for b in range(0, n_full, batch):
            flush(pending_s[b:b + batch], pending_p[b:b + batch])
            done += batch
            if time.time() - last > log_every:
                acc.block_until_ready()
                last = time.time()
                el = last - t0
                print(f"  correlators: {done}/{total} ({100 * done / total:.1f}%), {el / 3600:.2f} h, "
                      f"about {el * (total - done) / done / 3600:.2f} h left", flush=True)
        pending_s, pending_p = pending_s[n_full:], pending_p[n_full:]
    if len(pending_s):
        flush(pending_s, pending_p)
    acc = np.asarray(acc) / counts
    print(f"  correlators done in {(time.time() - t0) / 3600:.2f} h", flush=True)
    return {name: nk.stats.statistics(jnp.asarray(acc[:, k]).reshape(shape)) for k, name in enumerate(names)}


def measure(vstate, hi, g, positions=None, H=None, transverse=True, batch=1024):
    """The observables of build_observables, in the same order and from the same samples, with the
    XY correlators computed by xy_correlators instead of one vstate.expect per group: each pair
    (i, j) is evaluated once and assigned to its graph-distance shell, its Euclidean-distance shell
    and the corner-path distances it lies on."""
    N = g.n_nodes
    if H is None:
        H = build_hamiltonian(hi, g)
    out = {"E": vstate.expect(H)}   # draws the samples that every observable below uses
    if transverse:
        out["Mx"] = vstate.expect(sum(sigmax(hi, i) for i in range(N)) / N)
        out["My"] = vstate.expect(sum(sigmay(hi, i) for i in range(N)) / N)
    out["Mz"] = vstate.expect(sum(sigmaz(hi, i) for i in range(N)) / N)
    out.update(xy_correlators(vstate, correlator_pair_groups(g, positions), batch=batch))
    for i in range(N):
        if transverse:
            out[f"Mx_{i}"] = vstate.expect(sigmax(hi, i))
            out[f"My_{i}"] = vstate.expect(sigmay(hi, i))
        out[f"Mz_{i}"] = vstate.expect(sigmaz(hi, i))
    return out


def load_vstate(vstate_file, metadata_file, hi):
    with open(metadata_file) as f:
        metadata = json.load(f)
    model_cfg = metadata["model"]
    sampler_cfg = metadata["sampler"]

    model = nk.models.RBM(alpha=model_cfg.get("alpha", 1), kernel_init=nn.initializers.normal(stddev=0.01), hidden_bias_init=nn.initializers.normal(stddev=0.01))
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=sampler_cfg.get("n_chains", 32))
    vstate = nk.vqs.MCState(
        sampler,
        model,
        n_samples=sampler_cfg.get("n_samples", 2**16),
        n_discard_per_chain=sampler_cfg.get("n_discard_per_chain", 2**7),
        seed=sampler_cfg.get("seed", None),
    )
    with open(vstate_file, "rb") as f:
        return flax.serialization.from_bytes(vstate, f.read())


def _vit_mcstate(metadata_file, hi, patches, distances, chunk_size):
    from trainViT import ViT

    with open(metadata_file) as f:
        metadata = json.load(f)
    model_cfg = metadata["model"]
    sampler_cfg = metadata["sampler"]

    model = ViT(
        num_layers=model_cfg["num_layers"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        patches=tuple(map(tuple, patches)),
        distances=tuple(map(tuple, distances)),
    )
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=sampler_cfg.get("n_chains", 512))
    return nk.vqs.MCState(
        sampler,
        model,
        n_samples=sampler_cfg.get("n_samples", 2048),
        n_discard_per_chain=sampler_cfg.get("n_discard_per_chain", 0),
        seed=0,
        chunk_size=chunk_size,
    )


def load_vstate_vit(vstate_file, metadata_file, hi, patches, distances, chunk_size=None):
    vstate = _vit_mcstate(metadata_file, hi, patches, distances, chunk_size)
    with open(vstate_file, "rb") as f:
        return flax.serialization.from_bytes(vstate, f.read())


def load_vstate_vit_symm(vstate_file, metadata_file, hi, g, patches, distances, character_index=0, chunk_size=None):
    """Load a ViT state that was saved after symmetry projection."""
    vstate = _vit_mcstate(metadata_file, hi, patches, distances, chunk_size)
    rep = nk.symmetry.canonical_representation(hilbert=hi, group=g.automorphisms())
    proj_vstate = rep.project(vstate, character_index=character_index)
    proj_vstate.chunk_size = vstate.chunk_size
    with open(vstate_file, "rb") as f:
        return flax.serialization.from_bytes(proj_vstate, f.read())


def load_vstate_gcnn(vstate_file, metadata_file, hi, g):
    with open(metadata_file) as f:
        metadata = json.load(f)
    model_cfg = metadata["model"]
    sampler_cfg = metadata["sampler"]

    model = nk.models.GCNN(
        symmetries=g.automorphisms(),
        layers=model_cfg["num_layers"],
        features=tuple(model_cfg["feature_dims"]),
    )
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=sampler_cfg.get("n_chains", 32))
    vstate = nk.vqs.MCState(
        sampler,
        model,
        n_samples=sampler_cfg.get("n_samples", 2**16),
        n_discard_per_chain=sampler_cfg.get("n_discard_per_chain", 2**7),
        seed=sampler_cfg.get("seed", 0),
        chunk_size=None,
    )
    with open(vstate_file, "rb") as f:
        return flax.serialization.from_bytes(vstate, f.read())


def compute_obs(vstate, obs_dict):
    return {name: vstate.expect(op) for name, op in obs_dict.items()}


def save_obs(obs_dict, filename):
    out = {
        name: {
            "mean": float(np.real(value.mean)),
            "variance": float(np.real(value.variance)),
            "error": float(np.real(value.error_of_mean)),
        }
        for name, value in obs_dict.items()
    }
    with open(filename, "w") as f:
        json.dump(out, f, indent=2)


def run_sector(shape, G, seed, root, n_samples, chunk_size, pair_batch=1024):
    """Observables of a state saved by trainViT_sz.py (the C3v-projected one if present).
    It lives in a fixed S^z sector, so Mx and My are left out (they are zero there)."""
    from trainViT import build_fractal, patch_distances
    from sector_fidelity import N_PATCH, load_retrained_state

    g, patches, x, y = build_fractal(shape, G, N_PATCH)
    distances = patch_distances(patches, x, y)
    vstate, hi, info = load_retrained_state(shape, G, seed, g, patches, distances, root,
                                            n_samples=n_samples, n_discard=256, chunk_size=chunk_size)
    out = info["source"].replace(".mpack", "_observables.json")
    save_obs(measure(vstate, hi, g, positions=np.column_stack((x, y)), transverse=False, batch=pair_batch), out)
    print(f"Saved observables of {info['source']} to {out}")


def run_projected(shape, G, seed, n_samples, chunk_size, pair_batch=1024):
    """Observables of an original data/ViT state (C3v-projected) restricted to the S^z sector
    of sector_fidelity.target_sz, for generations without a retrained trainViT_sz.py state."""
    from trainViT import build_fractal, patch_distances
    from sector_fidelity import N_PATCH, load_sector_state

    g, patches, x, y = build_fractal(shape, G, N_PATCH)
    distances = patch_distances(patches, x, y)
    vstate, hi, info = load_sector_state(shape, G, seed, g, patches, distances, symm=True,
                                         n_samples=n_samples, n_discard=256, chunk_size=chunk_size)
    out = f"data/ViT/{shape}/vit_{shape}_G={G}_seed={seed}_sector_observables.json"
    save_obs(measure(vstate, hi, g, positions=np.column_stack((x, y)), transverse=False, batch=pair_batch), out)
    print(f"Saved sector projection of {info['source']} to {out} "
          f"(weight kept {info['weight_kept']:.3f}, flipped {info['flipped']})")


if __name__ == "__main__":
    import argparse

    from trainViT import build_fractal, patch_distances

    p = argparse.ArgumentParser(description="Observables of saved ViT states.")
    p.add_argument("--shapes", nargs="+", default=["triangular", "honeycomb"])
    p.add_argument("--gens", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--sector", metavar="ROOT", help="states of trainViT_sz.py under ROOT (e.g. data/ViT_sz) "
                   "instead of the full-space states in data/ViT")
    p.add_argument("--project-sector", action="store_true",
                   help="project the C3v states of data/ViT onto their S^z sector and write "
                   "*_sector_observables.json next to them")
    p.add_argument("--n-samples", type=int, default=2**12)
    p.add_argument("--chunk-size", type=int, default=1024,
                   help="--sector and --project-sector only")
    p.add_argument("--pair-batch", type=int, default=1024,
                   help="swapped configurations per network call in the XY correlators (memory: times "
                        "|C3v| = 6 network evaluations for projected states)")
    args = p.parse_args()
    shapes, gens, seeds = args.shapes, args.gens, args.seeds

    for shape in shapes:
        for G in gens:
            for seed in seeds:
                if args.project_sector:
                    run_projected(shape, G, seed, args.n_samples, args.chunk_size, args.pair_batch)
                    continue
                if args.sector:
                    run_sector(shape, G, seed, args.sector, args.n_samples, args.chunk_size, args.pair_batch)
                    continue
                base = f"data/ViT/{shape}/vit_{shape}_G={G}_seed={seed}"
                metadata_file = f"{base}_metadata.json"

                with open(metadata_file) as f:
                    n = json.load(f)["fractal"]["n"]

                g, patches, x, y = build_fractal(shape, G, n)
                distances = patch_distances(patches, x, y)
                hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
                pos = np.column_stack((x, y))

                vstate = load_vstate_vit(f"{base}.mpack", metadata_file, hi, patches, distances)
                vstate.n_samples = 2**12
                save_obs(measure(vstate, hi, g, positions=pos, batch=args.pair_batch), f"{base}_observables.json")

                proj_vstate = load_vstate_vit_symm(f"{base}_symm.mpack", metadata_file, hi, g, patches, distances)
                proj_vstate.n_samples = 2**12
                save_obs(measure(proj_vstate, hi, g, positions=pos, batch=args.pair_batch),
                         f"{base}_symm_observables.json")
                print(f"Saved observables for {base}")
