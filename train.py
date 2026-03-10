"""XY model simulation on fractal and regular lattices using NetKet."""

import numpy as np
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax
import jax.random as jr
import netket as nk
from netket.operator.spin import sigmax, sigmay, sigmaz
import flax
from flax import linen as nn
import optax
import time

from utils import *
from gaskets import *

print(f"NetKet {nk.__version__}, Jax {jax.__version__}")
print(f"JAX is using: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")


def save_vstate(vstate, filename):
    with open(filename, 'wb') as file:
        file.write(flax.serialization.to_bytes(vstate))

def save_metadata(filename, model_config, sampler_config, hamiltonian_config, final_results=None):
    """Save metadata about model, sampler, and Hamiltonian to JSON."""
    metadata = {
        "model": model_config,
        "sampler": sampler_config,
        "hamiltonian": hamiltonian_config
    }
    if final_results is not None:
        metadata["final_results"] = final_results
    with open(filename, 'w') as f:
        json.dump(metadata, f, indent=2)

def build_hamiltonian(hi, g, J=1.0, Jz=0.0, hx=0.0, hy = 0.0):
    H = nk.operator.LocalOperator(hi, dtype=complex)
    for i, j in g.edges():
        H += -J * (sigmax(hi, i) @ sigmax(hi, j) + sigmay(hi, i) @ sigmay(hi, j))
        if Jz != 0.0:
            H += -Jz * (sigmaz(hi, i) @ sigmaz(hi, j))
    if hx != 0.0 or hy != 0.0:
        for i in range(g.n_nodes):
            H += -hx * sigmax(hi, i) -hy * sigmay(hi, i)
    return H

def build_magnetizations(hi, g):
    N = g.n_nodes
    Mx = sum(sigmax(hi, i) for i in range(N)) / N
    My = sum(sigmay(hi, i) for i in range(N)) / N
    Mz = sum(sigmaz(hi, i) for i in range(N)) / N
    
    return {
        "Mx": Mx, "My": My, "Mz": Mz,
        "M2_perp": Mx @ Mx + My @ My,
    }

def run_xy_model(g, name, J=1.0, Jz=0.0, hx=None, hy=None, steps=250, alpha = 1, seed=None, n_samples = 2**16, n_chains=32, outfile = "vstate.mpack", compute_obs=False):
    start = time.time()
    
    if hx is None or hy is None:
        key = jr.PRNGKey(seed)
        key1, key2 = jr.split(key)
        if hx is None:
            hx = 0.01 * jr.uniform(key1, minval=-1.0, maxval=1.0)
        if hy is None:
            hy = 0.01 * jr.uniform(key2, minval=-1.0, maxval=1.0)
    
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
    H = build_hamiltonian(hi, g, J, Jz, hx, hy)
    
    if g.n_nodes < 16:
        E, _  = nk.exact.lanczos_ed(H, compute_eigenvectors=True, k=3)
        E_0 = E[0]
        E_1 = E[1]
        E_2 = E[2]
        gap1 = E_1 - E_0
        gap2 = E_2 - E_0
        print(f"{name} ED gaps: E1-E0={gap1}, E2-E0={gap2}")
        print(f"{name} ED ground state: {E_0}")
        

    obs_dict = None

    if compute_obs:
        obs_dict = build_magnetizations(hi, g)
    
    
    # model = nk.models.RBMModPhase(alpha=alpha, use_hidden_bias=True, kernel_init=nn.initializers.normal(stddev=0.01), hidden_bias_init=nn.initializers.normal(stddev=0.01),)
    model = nk.models.RBM(alpha=alpha, kernel_init=nn.initializers.normal(stddev=0.01), hidden_bias_init=nn.initializers.normal(stddev=0.01))
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=n_chains)
    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain= 2**7, seed=seed)
    
    # lr = optax.exponential_decay(init_value=0.01, transition_steps=100, decay_rate=0.9, end_value=1e-4)
    lr = 0.01
    optimizer = nk.optimizer.Sgd(learning_rate=lr)
    
    gs = nk.driver.VMC_SR(hamiltonian=H, optimizer=optimizer, variational_state=vstate, diag_shift=0.01)   
    log_file = f"{name}" if compute_obs else None
    gs.run(n_iter=steps, obs=obs_dict, out=log_file)
    
    # Print final energy and variance (σ² = <H²> - <H>²) after optimization
    # !!!!!
    # Can be costly for fourth generation
    E = vstate.expect(H)
    # E2_op = H @ H
    # E2 = vstate.expect(E2_op)
    # Evar = E2.mean - E.mean * E.mean
    Evar = E.variance
    print(f"E: {np.real(E.mean)}, σ²: {np.real(Evar)}")

    end = time.time()
    
    print("NQS parameters: ", vstate.n_parameters)
    print("Training time: ", end - start, " seconds")
    
    save_vstate(vstate, outfile)
    
    # Remove duplicate vstate file created by NetKet logger if it exists
    if compute_obs:
        log_mpack = f"{name}.log.mpack"
        if os.path.exists(log_mpack):
            os.remove(log_mpack)
            # print(f"Removed duplicate vstate file: {log_mpack}")
    
    # Save metadata
    metadata_file = outfile.replace('.mpack', '_metadata.json')
    model_config = {
        "type": "RBM",
        "alpha": alpha,
        "kernel_init": "normal(stddev=0.01)",
        "hidden_bias_init": "normal(stddev=0.01)"
    }
    sampler_config = {
        "type": "MetropolisLocal",
        "n_chains": n_chains,
        "n_samples": n_samples,
        "n_discard_per_chain": 2**7,
        "seed": seed
    }
    hamiltonian_config = {
        "J": J,
        "Jz": Jz,
        "hx": float(hx),
        "hy": float(hy)
    }
    # Store final energy and variance
    final_results = {
        "final_energy": float(np.real(E.mean)),
        "final_variance": float(np.real(Evar)),
        "training_time": end - start,
        "n_parameters": vstate.n_parameters
    }
    save_metadata(metadata_file, model_config, sampler_config, hamiltonian_config, final_results)
    print(f"Saved metadata to {metadata_file}")

    return vstate


if __name__ == "__main__":
    generations = [1, 2, 3, 4]
    seeds = [1, 2, 3, 4, 5]
    shapes = ['triangular', 'honeycomb']
    for shape in shapes:
        for (i, G) in enumerate(generations):
            for (j, seed) in enumerate(seeds):
                print(f"\n{'='*60}")
                print(f"Running generation {G}")
                print(f"{'='*60}\n")
                
                if shape == 'triangular':
                    A = t_gasket(G)
                else:
                    A = h_gasket(G)

                A = A.tocoo()   
                edges = list(zip(A.row, A.col))
                edges_list = [(i, j) for i, j in edges if i < j]
                
                name = f"{shape}_gasket_G={G}_seed={seed}"
                g_sierpinski = nk.graph.Graph(edges_list)
                print(f"{shape} gasket G={G}: {g_sierpinski.n_nodes} nodes\n")                        
                vstate = run_xy_model(g_sierpinski, name, steps=500, alpha=1, seed=seed, outfile=f"{name}.mpack", compute_obs=True, n_samples=2**17, n_chains=32, hy=0, hx=0)

