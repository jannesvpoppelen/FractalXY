
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import json
import jax
import jax.random as jr
import netket as nk
from netket.operator.spin import sigmax, sigmay, sigmaz
import flax
from flax import linen as nn

from utils import *
from fractals import *


def build_observables(hi, g, positions=None):
    N = g.n_nodes
    H = build_hamiltonian(hi, g)
    
    Mx = sum(sigmax(hi, i) for i in range(N)) / N
    My = sum(sigmay(hi, i) for i in range(N)) / N
    Mz = sum(sigmaz(hi, i) for i in range(N)) / N
    
    obs_dict = {
        "E": H,
        "Mx": Mx,
        "My": My,
        "Mz": Mz,
        # "M2_perp": Mx @ Mx + My @ My,
    }
    
    # All-to-all graph distance correlators, includes bulk
    shells = get_distance_shells(g)
    for d, pairs in shells.items():
        print(f"Graph distance shell d={d}: {len(pairs)} pairs")
        op = sum(sigmax(hi, i) @ sigmax(hi, j) + sigmay(hi, i) @ sigmay(hi, j) 
                 for i, j in pairs) / len(pairs)
        obs_dict[f"Cxy_graph_r{d}"] = op
    
    # Corner path correlators
    corner_obs = build_corner_path_correlators(hi, g)
    obs_dict.update(corner_obs)

    if positions is not None:
        # Euclidean distance correlators (all-to-all)
        euclidean_shells = get_euclidean_distance_shells_all(positions)
        for d, pairs in sorted(euclidean_shells.items()):
            print(f"Euclidean distance shell d={d:.4f}: {len(pairs)} pairs")
            op = sum(sigmax(hi, i) @ sigmax(hi, j) + sigmay(hi, i) @ sigmay(hi, j) 
                     for i, j in pairs) / len(pairs)
            obs_dict[f"Cxy_euclidean_r{d:.4f}"] = op
    
    # Site-resolved magnetizations
    Mx_total = sum(sigmax(hi, j) for j in range(N)) / N
    My_total = sum(sigmay(hi, j) for j in range(N)) / N
    for i in range(N):
        obs_dict[f"Mx_{i}"] = sigmax(hi, i)
        obs_dict[f"My_{i}"] = sigmay(hi, i)
        obs_dict[f"Mz_{i}"] = sigmaz(hi, i)
        #obs_dict[f"M2_perp_{i}"] = sigmax(hi, i) @ Mx_total + sigmay(hi, i) @ My_total
    
    return obs_dict


def load_vstate(vstate_file, metadata_file, hi):
    
    with open(metadata_file, 'r') as metafile:
        metadata = json.load(metafile)
        model_config = metadata["model"]
        sampler_config = metadata["sampler"]
        hamiltonian_config = metadata["hamiltonian"]
        
        
    alpha = model_config.get("alpha", 1)
    n_chains = sampler_config.get("n_chains", 32)
    n_samples = sampler_config.get("n_samples", 2**16)
    n_discard_per_chain = sampler_config.get("n_discard_per_chain", 2**7)
    seed = sampler_config.get("seed", None)
        
    model = nk.models.RBM(alpha=alpha, kernel_init=nn.initializers.normal(stddev=0.01), hidden_bias_init=nn.initializers.normal(stddev=0.01))
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=n_chains)
    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain, seed=seed)
    
    with open(vstate_file, 'rb') as vstatefile:
        vstate = flax.serialization.from_bytes(vstate, vstatefile.read())
    return vstate


def load_vstate_vit(vstate_file, metadata_file, hi, patches, distances):
    from trainViT import ViT

    with open(metadata_file) as f:
        metadata = json.load(f)

    model_cfg = metadata["model"]
    sampler_cfg = metadata["sampler"]

    patches_tuple = tuple(map(tuple, patches))
    distances_tuple = tuple(map(tuple, distances))

    model = ViT(
        num_layers=model_cfg["num_layers"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        patches=patches_tuple,
        distances=distances_tuple,
    )
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=sampler_cfg.get("n_chains", 512))
    vstate = nk.vqs.MCState(
        sampler,
        model,
        n_samples=sampler_cfg.get("n_samples", 2048),
        n_discard_per_chain=sampler_cfg.get("n_discard_per_chain", 0),
        seed=0,
        chunk_size=128,
    )

    with open(vstate_file, "rb") as f:
        vstate = flax.serialization.from_bytes(vstate, f.read())

    return vstate


def load_vstate_vit_symm(vstate_file, metadata_file, hi, g, patches, distances, character_index=0):
    """Load a symmetry-projected ViT vstate saved via rep.project()."""
    from trainViT import ViT

    with open(metadata_file) as f:
        metadata = json.load(f)

    model_cfg = metadata["model"]
    sampler_cfg = metadata["sampler"]

    patches_tuple = tuple(map(tuple, patches))
    distances_tuple = tuple(map(tuple, distances))

    model = ViT(
        num_layers=model_cfg["num_layers"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        patches=patches_tuple,
        distances=distances_tuple,
    )
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=sampler_cfg.get("n_chains", 512))
    vstate = nk.vqs.MCState(
        sampler,
        model,
        n_samples=sampler_cfg.get("n_samples", 2048),
        n_discard_per_chain=sampler_cfg.get("n_discard_per_chain", 0),
        seed=0,
        chunk_size=128,
    )

    group = g.automorphisms()
    rep = nk.symmetry.canonical_representation(hilbert=hi, group=group)
    proj_vstate = rep.project(vstate, character_index=character_index)
    proj_vstate.chunk_size = vstate.chunk_size

    with open(vstate_file, "rb") as f:
        proj_vstate = flax.serialization.from_bytes(proj_vstate, f.read())

    return proj_vstate


def compute_obs(vstate, obs_dict):
    results = {}
    for name, op in obs_dict.items():
        results[name] = vstate.expect(op)
    return results


def save_obs(obs_dict, filename):
    serializable_dict = {}
    for name, value in obs_dict.items():
        serializable_dict[name] = {
            "mean": float(np.real(value.mean)),
            "variance": float(np.real(value.variance)),
            "error": float(np.real(value.error_of_mean))
        }
    with open(filename, 'w') as f:
        json.dump(serializable_dict, f, indent=2)
    


if __name__ == "__main__":
    gens = [4]
    seeds = [1, 2, 3, 4, 5]
    #shapes = ['triangular', 'honeycomb']
    shapes = ['honeycomb']

    for shape in shapes:
        for (i, G) in enumerate(gens):
            if shape == 'triangular':
                A = t_gasket(G)
            else:
                A = h_gasket(G)
            A = A.tocoo()
            edges = list(zip(A.row, A.col))
            edges_list = [(i, j) for i, j in edges if i < j]
            x, y  = gasket_coordinates(G)
            vertices = np.column_stack((x, y))
            g_sierpinski = nk.graph.Graph(edges_list)
            hi = nk.hilbert.Spin(s=0.5, N=g_sierpinski.n_nodes)
            obs_dict = build_observables(hi, g_sierpinski, positions=vertices)
            for (j, seed) in enumerate(seeds):
                vstate = load_vstate(f"data/{shape}/{shape}_gasket_G={G}_seed={seed}.mpack", f"data/{shape}/{shape}_gasket_G={G}_seed={seed}_metadata.json", hi)
                vstate.chunk_size = 512
                results = compute_obs(vstate, obs_dict)
                save_obs(results, f"data/{shape}/{shape}_gasket_G={G}_seed={seed}_observables.json")
