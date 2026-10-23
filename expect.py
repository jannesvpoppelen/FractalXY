import json

import numpy as np
import flax
import netket as nk
from flax import linen as nn
from netket.operator.spin import sigmax, sigmay, sigmaz

from utils import build_hamiltonian, get_distance_shells, get_euclidean_distance_shells_all, build_corner_path_correlators


def xy_correlator(hi, pairs):
    return sum(sigmax(hi, i) @ sigmax(hi, j) + sigmay(hi, i) @ sigmay(hi, j) for i, j in pairs) / len(pairs)


def build_observables(hi, g, positions=None, H=None):
    N = g.n_nodes
    if H is None:
        H = build_hamiltonian(hi, g)

    obs_dict = {
        "E": H,
        "Mx": sum(sigmax(hi, i) for i in range(N)) / N,
        "My": sum(sigmay(hi, i) for i in range(N)) / N,
        "Mz": sum(sigmaz(hi, i) for i in range(N)) / N,
    }

    for d, pairs in get_distance_shells(g).items():
        print(f"Graph distance shell d={d}: {len(pairs)} pairs")
        obs_dict[f"Cxy_graph_r{d}"] = xy_correlator(hi, pairs)

    if positions is not None:
        obs_dict.update(build_corner_path_correlators(hi, g, positions))
        for d, pairs in sorted(get_euclidean_distance_shells_all(positions).items()):
            print(f"Euclidean distance shell d={d:.4f}: {len(pairs)} pairs")
            obs_dict[f"Cxy_euclidean_r{d:.4f}"] = xy_correlator(hi, pairs)

    for i in range(N):
        obs_dict[f"Mx_{i}"] = sigmax(hi, i)
        obs_dict[f"My_{i}"] = sigmay(hi, i)
        obs_dict[f"Mz_{i}"] = sigmaz(hi, i)

    return obs_dict


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


if __name__ == "__main__":
    from trainViT import build_fractal, patch_distances

    shapes = ["triangular", "honeycomb"]
    gens = [1, 2, 3, 4, 5]
    seeds = [1, 2, 3, 4, 5]

    for shape in shapes:
        for G in gens:
            for seed in seeds:
                base = f"data/ViT/{shape}/vit_{shape}_G={G}_seed={seed}"
                metadata_file = f"{base}_metadata.json"

                with open(metadata_file) as f:
                    n = json.load(f)["fractal"]["n"]

                g, patches, x, y = build_fractal(shape, G, n)
                distances = patch_distances(patches, x, y)
                hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
                obs_dict = build_observables(hi, g, positions=np.column_stack((x, y)))

                vstate = load_vstate_vit(f"{base}.mpack", metadata_file, hi, patches, distances)
                vstate.n_samples = 2**12
                save_obs(compute_obs(vstate, obs_dict), f"{base}_observables.json")

                proj_vstate = load_vstate_vit_symm(f"{base}_symm.mpack", metadata_file, hi, g, patches, distances)
                proj_vstate.n_samples = 2**12
                save_obs(compute_obs(proj_vstate, obs_dict), f"{base}_symm_observables.json")
                print(f"Saved observables for {base}")
