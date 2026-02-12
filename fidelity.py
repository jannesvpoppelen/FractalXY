import numpy as np
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax
import netket as nk
from netket.operator.spin import sigmax, sigmay, sigmaz
from netket.experimental.observable import InfidelityOperator
import flax
from flax import linen as nn
from utils import *

print(f"NetKet {nk.__version__}, Jax {jax.__version__}")
print(f"JAX is using: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")


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


def compute_fidelity(vstate1, vstate2):
# Use fidelity estimator to compare different variational states
    obs = InfidelityOperator(vstate1)
    result = vstate2.expect(obs)
    fidelity = 1.0 - result.mean.item()
    return fidelity


if __name__ == "__main__":
    generations = [1, 2, 3, 4]
    seeds = [1, 2, 3, 4, 5]

    for gen in generations:
        print(f"\n{'='*60}")
        print(f"Fidelity comparisons for generation {gen}")
        print(f"{'='*60}\n")
        
        edges = np.genfromtxt(f"edges{gen}.txt", dtype=int)
        edges_list = [tuple(map(int, e)) for e in edges]
        vertices = np.genfromtxt(f"vertices{gen}.txt")
        
        g_sierpinski = nk.graph.Graph(edges_list)
        hi = nk.hilbert.Spin(s=0.5, N=g_sierpinski.n_nodes)
        print(f"Sierpinski gen{gen}: {g_sierpinski.n_nodes} nodes\n")
        
        vstates = []
        for seed in seeds:
            vstate_file = f"data/sierpinski_gen{gen}_seed{seed}.mpack"
            metadata_file = f"data/sierpinski_gen{gen}_seed{seed}_metadata.json"
            vstate = load_vstate(vstate_file, metadata_file, hi)
            vstates.append(vstate)
        
        fidelities = {}
        for i in range(len(seeds)):
            for j in range(i+1, len(seeds)):
                fid = compute_fidelity(vstates[i], vstates[j])
                fidelities[f"seed{seeds[i]}_seed{seeds[j]}"] = float(fid)
                print(f"Fidelity between seed {seeds[i]} and seed {seeds[j]}: {fid}")
        
        # Store results for this generation
        gen_results = {
            "generation": gen,
            "seeds": seeds,
            "fidelities": fidelities,
            "mean_fidelity": float(np.mean(list(fidelities.values()))),
            "std_fidelity": float(np.std(list(fidelities.values()))),
            "min_fidelity": float(np.min(list(fidelities.values()))),
            "max_fidelity": float(np.max(list(fidelities.values())))
        }
        
        print(f"\nGen {gen} summary: mean fidelity = {gen_results['mean_fidelity']:.4f} ± {gen_results['std_fidelity']:.4f}")

        filename = f"fidelity_gen{gen}.json"
        with open(filename, "w") as f:
            json.dump(gen_results, f, indent=2)
        print(f"Fidelity results saved to {filename}")
