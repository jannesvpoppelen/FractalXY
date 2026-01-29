
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
from train import build_hamiltonian


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
        "M2_perp": Mx @ Mx + My @ My,
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
        obs_dict[f"M2_perp_{i}"] = sigmax(hi, i) @ Mx_total + sigmay(hi, i) @ My_total
    
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
        }
    with open(filename, 'w') as f:
        json.dump(serializable_dict, f, indent=2)
    


if __name__ == "__main__":
    gens = [1]
    seeds = [1, 2, 3, 4]

    for gen in gens:
        edges = np.genfromtxt(f"edges{gen}.txt", dtype=int)
        edges_list = [tuple(map(int, e)) for e in edges]
        vertices = np.genfromtxt(f"vertices{gen}.txt")
        g_sierpinski = nk.graph.Graph(edges_list)
        hi = nk.hilbert.Spin(s=0.5, N=g_sierpinski.n_nodes)
        obs_dict = build_observables(hi, g_sierpinski, positions=vertices)
        for seed in seeds:
            vstate = load_vstate(f"sierpinski_gen{gen}_seed{seed}.mpack", f"sierpinski_gen{gen}_seed{seed}_metadata.json", hi)
            results = compute_obs(vstate, obs_dict)
            save_obs(results, f"sierpinski_gen{gen}_seed{seed}_observables.json")
