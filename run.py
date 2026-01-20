"""XY model simulation on fractal and regular lattices using NetKet."""

import numpy as np
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax
import netket as nk
from netket.operator.spin import sigmax, sigmay, sigmaz
from netket.experimental.observable import InfidelityOperator
from flax import linen as nn
import optax

from utils import *

print(f"NetKet {nk.__version__}, Jax {jax.__version__}")
print(f"JAX is using: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")


def build_hamiltonian(hi, g, J=1.0, Jz=0.0, h=0.0):
    H = nk.operator.LocalOperator(hi, dtype=complex)
    for i, j in g.edges():
        H += -J * (sigmax(hi, i) @ sigmax(hi, j) + sigmay(hi, i) @ sigmay(hi, j))
        if Jz != 0.0:
            H += -Jz * (sigmaz(hi, i) @ sigmaz(hi, j))
    if h != 0.0:
        for i in range(g.n_nodes):
            H += -h * (sigmay(hi, i) + sigmax(hi, i))
    return H


def build_basic_observables(hi, g):
    N = g.n_nodes
    Mx = sum(sigmax(hi, i) for i in range(N)) / N
    My = sum(sigmay(hi, i) for i in range(N)) / N
    Mz = sum(sigmaz(hi, i) for i in range(N)) / N
    
    return {
        "Mx": Mx, "My": My, "Mz": Mz,
        "M2_perp": Mx @ Mx + My @ My,
    }


def build_full_observables(hi, g, positions=None, name=""):
    N = g.n_nodes
    obs_dict = build_basic_observables(hi, g)
    
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
        # Structure factors
        k_list = get_k_vectors(name, positions)
        obs_dict.update(build_structure_factor_ops(hi, positions, k_list))
    
    # Site-resolved magnetizations
    Mx_total = sum(sigmax(hi, j) for j in range(N)) / N
    My_total = sum(sigmay(hi, j) for j in range(N)) / N
    for i in range(N):
        obs_dict[f"Mx_{i}"] = sigmax(hi, i)
        obs_dict[f"My_{i}"] = sigmay(hi, i)
        obs_dict[f"Mz_{i}"] = sigmaz(hi, i)
        obs_dict[f"M2_perp_{i}"] = sigmax(hi, i) @ Mx_total + sigmay(hi, i) @ My_total
    
    return obs_dict



def run_xy_model(g, name, positions=None, J=1.0, Jz=0.0, h=0.0, steps=250, alpha = 1, seed=None, compute_obs=True):
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
    H = build_hamiltonian(hi, g, J, Jz, h)
    ntk = 0 
    otf = 0
    # Exact diagonalization for first 3 states for small systems
    if g.n_nodes < 16:
        E_gs, psi_gs  = nk.exact.lanczos_ed(H, compute_eigenvectors=True, k=3)
        E_0 = E_gs[0]
        E_1 = E_gs[1]
        E_2 = E_gs[2]
        gap1 = E_1 - E_0
        gap2 = E_2 - E_0
        print(f"{name} ED gaps: E1-E0={gap1}, E2-E0={gap2}")
        print(f"{name} ED ground state: {E_gs}")
    else:
        E_gs = None
    
    basic_obs = build_basic_observables(hi, g)
    # model = nk.models.RBMModPhase(alpha=alpha, use_hidden_bias=True, kernel_init=nn.initializers.normal(stddev=0.01), hidden_bias_init=nn.initializers.normal(stddev=0.01),)
    model = nk.models.RBM(alpha=alpha, kernel_init=nn.initializers.normal(stddev=0.01), hidden_bias_init=nn.initializers.normal(stddev=0.01))
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=1024)
    
    if g.n_nodes > 100:
        sampler = nk.sampler.MetropolisLocal(hi, n_chains=256)

    vstate = nk.vqs.MCState(sampler, model, n_samples=2**13, n_discard_per_chain=8, chunk_size = None, seed=seed)
    lr = optax.linear_schedule(0.01, 0.0001, 500)
    lr = optax.exponential_decay(init_value=0.01,transition_steps=100,decay_rate=0.9,end_value=1e-4,)
    optimizer = nk.optimizer.Sgd(learning_rate=lr)
    if g.n_nodes>25:
        vstate.n_samples = 2**13
        vstate.chunk_size = 1024
        if alpha > 1:
            vstate.n_samples = 2**12
    
    if g.n_nodes>100:
        ntk = 1
        otf = 1
        vstate.n_samples = 2**10
    
    gs = nk.driver.VMC_SR(hamiltonian=H, optimizer=optimizer, variational_state=vstate, diag_shift=0.001, use_ntk = ntk, on_the_fly = otf)
   
    if compute_obs:
        gs.run(n_iter=steps, out=name, obs=basic_obs, write_every=5)
    else:
        gs.run(n_iter=steps)
    
    # Print final energy and physical variance (σ² = <H²> - <H>²) after optimization
    E = vstate.expect(H)
    Evar_op = nk.experimental.observable.VarianceObservable(H)
    Evar = vstate.expect(Evar_op)
    print(f"Final energy: {np.real(E.mean)}, σ² = {np.real(Evar.mean)}")
    
    # Compute full observables after training
    if not compute_obs:
        return vstate
    
    print(f"Computing observables...")
    full_obs = build_full_observables(hi, g, positions, name)
    full_obs["E"] = H
    
    results = {}
    for key, op in full_obs.items():
        result = vstate.expect(op)
        results[key] = {
            "mean": float(np.real(result.mean.item())),
            "variance": float(np.real(result.variance.item()))
        }

    # Add E_gs to results if available
    if E_gs is not None:
        results["E_gs"] = E_0

    # If E_gs is not None, compute exact total magnetizations and in plane fluctuations
    if E_gs is not None:
        psi_gs = psi_gs[:,0]
        N = g.n_nodes
        Mx_exact = sum(psi_gs.conj().T @ sigmax(hi, i).to_dense() @ psi_gs for i in range(N)) / N
        My_exact = sum(psi_gs.conj().T @ sigmay(hi, i).to_dense() @ psi_gs for i in range(N)) / N
        Mz_exact = sum(psi_gs.conj().T @ sigmaz(hi, i).to_dense() @ psi_gs for i in range(N)) / N
        M2_perp_exact = psi_gs.conj().T @ (sum(sigmax(hi, i) for i in range(N)) / N).to_dense() @ (sum(sigmax(hi, j) for j in range(N)) / N).to_dense() @ psi_gs + \
                         psi_gs.conj().T @ (sum(sigmay(hi, i) for i in range(N)) / N).to_dense() @ (sum(sigmay(hi, j) for j in range(N)) / N).to_dense() @ psi_gs

        results["Mx_exact"] = float(np.real(Mx_exact))
        results["My_exact"] = float(np.real(My_exact))
        results["Mz_exact"] = float(np.real(Mz_exact))
        results["M2_perp_exact"] = float(np.real(M2_perp_exact))



    with open(f"{name}_observables.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Observables saved to {name}_observables.json")
    
    return vstate


generations = [1, 2, 3, 4]
alphas = [1, 2]

for gen in generations:
    for alpha in alphas:
        print(f"\n{'='*60}")
        print(f"Running generation {gen}")
        print(f"{'='*60}\n")
        
        edges = np.genfromtxt(f"edges{gen}.txt", dtype=int)
        edges_list = [tuple(map(int, e)) for e in edges]
        vertices = np.genfromtxt(f"vertices{gen}.txt")
        
        g_sierpinski = nk.graph.Graph(edges_list)
        print(f"Sierpinski gen{gen}: {g_sierpinski.n_nodes} nodes\n")
        
        vstate = run_xy_model(g_sierpinski, f"sierpinski_gen{gen}_alpha{alpha}", 
                            positions=vertices, steps=500, alpha=alpha, h = 0.0)

# ============================================================== #

def compute_fidelity(vstate1, vstate2):
# Use fidelity estimator to compare different variational states
    obs = InfidelityOperator(vstate1)
    result = vstate2.expect(obs)
    fidelity = 1.0 - result.mean.item()
    return fidelity

# Do VMC for different intial seeds and compare final states
generations = [1, 2, 3, 4]
seeds = [1, 2, 3, 5, 8]

all_results = {}


for gen in generations:
    print(f"\n{'='*60}")
    print(f"Fidelity comparisons for generation {gen}")
    print(f"{'='*60}\n")
    
    edges = np.genfromtxt(f"edges{gen}.txt", dtype=int)
    edges_list = [tuple(map(int, e)) for e in edges]
    vertices = np.genfromtxt(f"vertices{gen}.txt")
    
    g_sierpinski = nk.graph.Graph(edges_list)
    print(f"Sierpinski gen{gen}: {g_sierpinski.n_nodes} nodes\n")
    
    vstates = []
    for seed in seeds:
        vstate = run_xy_model(g_sierpinski, f"sierpinski_gen{gen}_seed{seed}", 
                            positions=vertices, steps=500, alpha=1, seed=seed, compute_obs=False)
        vstates.append(vstate)
    
    fidelities = {}
    for i in range(len(seeds)):
        for j in range(i+1, len(seeds)):
            fid = compute_fidelity(vstates[i], vstates[j])
            fidelities[f"seed{seeds[i]}_seed{seeds[j]}"] = float(fid)
            print(f"Fidelity between seed {seeds[i]} and seed {seeds[j]}: {fid}")
    
    # Store results for this generation
    all_results[f"gen{gen}"] = {
        "seeds": seeds,
        "fidelities": fidelities,
        "mean_fidelity": float(np.mean(list(fidelities.values()))),
        "std_fidelity": float(np.std(list(fidelities.values()))),
        "min_fidelity": float(np.min(list(fidelities.values()))),
        "max_fidelity": float(np.max(list(fidelities.values())))
    }
    
    print(f"\nGen {gen} summary: mean fidelity = {all_results[f'gen{gen}']['mean_fidelity']:.4f} ± {all_results[f'gen{gen}']['std_fidelity']:.4f}")

# Save all results to file
with open("fidelity_analysis.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nAll fidelity results saved to fidelity_analysis.json")
