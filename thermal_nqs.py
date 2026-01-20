"""
Purified thermal state simulations using NetKet and RBM ansatz.

Implements the thermofield double formalism for finite-temperature quantum states.
"""

import json
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax
import netket as nk
from netket.operator.spin import sigmax, sigmay, sigmaz
import netket.experimental as nkx
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt

print("NetKet version:", nk.__version__)
print("Jax version:", jax.__version__)
print(jax.devices())


# ---------------------------------------------------------------------------
# Custom RBM for purified thermal states
# ---------------------------------------------------------------------------

class RBM(eqx.Module):
    """
    RBM for purified thermal states.
    log ψ(σ, s) = Σ_i a_i σ_i + a'_i s_i + Σ_m log cosh(b_m + Σ_j W_mj σ_j + W'_mj s_j)

    σ : visible spins (real system)
    s : ancilla spins (purification)
    """
    a: jax.Array
    aprime: jax.Array
    b: jax.Array
    W: jax.Array
    Wprime: jax.Array
    
    def __init__(self, n_visible, n_hidden, *, key=None, beta0_init=True):
        if beta0_init:
            # β=0: W = iπ/4 δ_ij, W' = -iπ/4 δ_ij (maximally entangled Bell pairs)
            self.a = jnp.zeros(n_visible)
            self.aprime = jnp.zeros(n_visible)
            self.b = jnp.zeros(n_hidden)
            self.W = 1j * (jnp.pi / 4) * jnp.eye(n_visible, n_visible)[:n_hidden, :]
            self.Wprime = -1j * (jnp.pi / 4) * jnp.eye(n_visible, n_visible)[:n_hidden, :]

            # Add some small random noise to break symmetry
            self.W += 1e-2 * jax.random.normal(jax.random.PRNGKey(42), self.W.shape)
            self.Wprime += 1e-2 * jax.random.normal(jax.random.PRNGKey(43), self.Wprime.shape)
        else:
            if key is None:
                key = jax.random.PRNGKey(0)
            key_a, key_a_prime, key_b, key_W, key_Wprime = jax.random.split(key, 5)
            self.a = jax.random.normal(key_a, (n_visible,)) * 0.01
            self.aprime = jax.random.normal(key_a_prime, (n_visible,)) * 0.01
            self.b = jax.random.normal(key_b, (n_hidden,)) * 0.01
            self.W = jax.random.normal(key_W, (n_hidden, n_visible)) * 0.01
            self.Wprime = jax.random.normal(key_Wprime, (n_hidden, n_visible)) * 0.01
    
    def __call__(self, sigma, **kwargs):
        n = self.a.shape[0]
        sigma_real = sigma[..., :n]
        sigma_ancilla = sigma[..., n:]
        
        visible_term = jnp.dot(sigma_real, self.a) + jnp.dot(sigma_ancilla, self.aprime)
        theta = self.b + jnp.dot(sigma_real, self.W.T) + jnp.dot(sigma_ancilla, self.Wprime.T)
        hidden_term = jnp.sum(jnp.log(jnp.cosh(theta)), axis=-1)
        
        return visible_term + hidden_term


# ---------------------------------------------------------------------------
# Hamiltonian builders
# ---------------------------------------------------------------------------

def build_xy_hamiltonian(hi_total, edges, N, J=1.0, Jz=0.0, h=0.0):
    """
    Build XY model Hamiltonian H = -J Σ_<ij> (σ^x_i σ^x_j + σ^y_i σ^y_j)
    Acting on real system only (H ⊗ I).
    """
    H = 0
    for i, j in edges:
        H += -J * (sigmax(hi_total, i) @ sigmax(hi_total, j) +
                   sigmay(hi_total, i) @ sigmay(hi_total, j))
        if Jz != 0.0:
            H += -Jz * (sigmaz(hi_total, i) @ sigmaz(hi_total, j))
    
    if h != 0.0:
        for i in range(N):
            H += -h * sigmaz(hi_total, i)
    
    return H


def build_tfim_hamiltonian(hi_total, edges, N, J=1.0, hx=1.0):
    """
    Build Transverse Field Ising Model: H = -J Σ_<ij> σ^z_i σ^z_j - hx Σ_i σ^x_i
    Acting on real system only (H ⊗ I).
    """
    H = 0
    # Ising interaction
    for i, j in edges:
        H += -J * (sigmaz(hi_total, i) @ sigmaz(hi_total, j))
    
    # Transverse field
    for i in range(N):
        H += -hx * sigmax(hi_total, i)
    
    return H


# ---------------------------------------------------------------------------
# Main simulation setup
# ---------------------------------------------------------------------------

def run_thermal_simulation(graph, model_type="xy", J=1.0, params={}, 
                          n_samples=2**18, n_chains=64, n_discard=2000, tdvp_time=1.0, dt=0.001):
    """
    Run purified thermal state simulation.
    
    Args:
        graph: NetKet graph for the lattice
        model_type: "xy" or "tfim"
        J: coupling strength
        params: additional model parameters (Jz, h for XY; hx for TFIM)
        n_samples, n_chains, n_discard: sampling parameters
        do_tdvp: whether to run imaginary time evolution
        tdvp_time: total imaginary time
        dt: time step for TDVP
    """
    N = graph.n_nodes
    edges = list(graph.edges())
    
    # Build doubled Hilbert space
    hi_real = nk.hilbert.Spin(s=0.5, N=N)
    hi_ancilla = nk.hilbert.Spin(s=0.5, N=N)
    hi_total = nk.hilbert.TensorHilbert(hi_real, hi_ancilla)
    
    # Build Hamiltonian
    if model_type == "xy":
        Jz = params.get("Jz", 0.0)
        h = params.get("h", 0.0)
        H = build_xy_hamiltonian(hi_total, edges, N, J=J, Jz=Jz, h=h)
    elif model_type == "tfim":
        hx = params.get("hx", 1.0)
        H = build_tfim_hamiltonian(hi_total, edges, N, J=J, hx=hx)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    print(f"\nModel: {model_type.upper()}")
    print(f"Hilbert space: {N} real + {N} ancilla = {hi_total.size} total")
    print(f"Parameters: J={J}, {params}")
    
    # Initialize RBM at β=0
    model = RBM(n_visible=N, n_hidden=N, beta0_init=True)
    
    print(f"\nInitialized RBM: {N} visible, {N} hidden units (α=1)")
    print(f"β=0 initialization (infinite temperature)")
    
    # Create variational state
    sampler = nk.sampler.MetropolisLocal(hi_total, n_chains=n_chains)
    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, 
                           n_discard_per_chain=n_discard)
    
    obs = {"sx": sum(nk.operator.spin.sigmax(hi, i) for i in range(N))}
    
    
    print(f"\nRunning imaginary-time TDVP (T={tdvp_time}, dt={dt})...")
    log = nk.logging.RuntimeLog()
    integrator = nkx.dynamics.Heun(dt=dt)
    qgt = nk.optimizer.qgt.QGTJacobianDense(holomorphic=True)
    te = nkx.TDVP(H, vstate, integrator, qgt=qgt, propagation_type='imag')
    te.run(T=tdvp_time, out=log, obs =obs)

    
    return vstate, H, obs


# ---------------------------------------------------------------------------
# Temperature sweep
# ---------------------------------------------------------------------------

def temperature_sweep(graph, model_type="xy", J=1.0, params={},
                     T_min=0.1, T_max=10.0, n_temps=10,
                     n_samples=2**16, n_chains=64, dt=0.001,
                     output_file=None):
    """
    Compute energy per site at different temperatures using imaginary-time TDVP.
    
    For purified states: β = 2τ, where τ is the TDVP imaginary time parameter.
    To reach temperature T, we need β = 1/T, so τ = 1/(2T).
    
    Args:
        graph: NetKet graph
        model_type: "xy" or "tfim"
        J: coupling strength
        params: model parameters
        T_min, T_max: temperature range
        n_temps: number of temperature points
        n_samples, n_chains: sampling parameters
        dt: TDVP time step
        output_file: optional file to save results
        
    Returns:
        temps: array of temperatures
        energies: array of energies per site
        errors: array of error estimates
    """
    N = graph.n_nodes
    edges = list(graph.edges())
    
    # Build doubled Hilbert space
    hi_real = nk.hilbert.Spin(s=0.5, N=N)
    hi_ancilla = nk.hilbert.Spin(s=0.5, N=N)
    hi_total = nk.hilbert.TensorHilbert(hi_real, hi_ancilla)
    
    # Build Hamiltonian
    if model_type == "xy":
        Jz = params.get("Jz", 0.0)
        h = params.get("h", 0.0)
        H = build_xy_hamiltonian(hi_total, edges, N, J=J, Jz=Jz, h=h)
    elif model_type == "tfim":
        hx = params.get("hx", 1.0)
        H = build_tfim_hamiltonian(hi_total, edges, N, J=J, hx=hx)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Initialize at β=0 (infinite temperature)
    model = RBM(n_visible=N, n_hidden=N, beta0_init=True)
    sampler = nk.sampler.MetropolisLocal(hi_total, n_chains=n_chains)
    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain=1000)
    
    # Setup TDVP with improved convergence
    integrator = nkx.dynamics.RK4(dt=dt)  # 4th order Runge-Kutta for better accuracy
    qgt = nk.optimizer.qgt.QGTJacobianDense(     # Use holomorphic gradients for complex parameters
        diag_shift=0.01        # Regularization to stabilize QGT inversion
    )
    te = nkx.TDVP(H, vstate, integrator, qgt=qgt, propagation_type='imag')
    
    # Temperature points (logarithmic spacing from high to low T)
    # temps = np.logspace(np.log10(T_max), np.log10(T_min), n_temps)
    temps = np.array([0.1, 0.4, 0.7, 0.9, 1, 1.05, 1.1, 1.5, 2, 2.5, 3, 4, 5, 7, 10])  # Manually defined for better coverage
    # Reverse temp order to go from high T to low T
    temps = temps[temps.argsort()[::-1]]
    betas = 1.0 / temps
    taus = betas / 2.0  # TDVP time parameter T = β/2
    
    energies = []
    errors = []
    
    print(f"\n{'='*70}")
    # print(f"Temperature sweep: {model_type.upper()} on {graph}")
    # print(f"T ∈ [{T_min:.2f}, {T_max:.2f}], {n_temps} points")
    print(f"Relation: β = 1/T, TDVP time T = β/2")
    print(f"Integrator: RK4, dt={dt}, QGT diag_shift=0.01")
    print(f"{'='*70}")
    print(f"{'T':>8s} {'β':>10s} {'τ(TDVP)':>12s} {'E/N':>12s} {'Error':>10s}")
    print(f"{'-'*70}")
    
    current_tau = 0.0  # Start at τ=0 (β=0, infinite temperature)
    
    for i, (T, beta, tau) in enumerate(zip(temps, betas, taus)):
        tau_step = tau - current_tau
        
        if tau_step > 0:
            # Run TDVP to reach target τ (equivalently, target β)
            te.run(T=tau_step, show_progress=True)
            current_tau = tau
        
        # Measure energy
        E_result = vstate.expect(H)
        E_per_site = E_result.mean.real / N
        E_error = E_result.error_of_mean.real / N if hasattr(E_result, 'error_of_mean') else 0.0
        
        energies.append(E_per_site)
        errors.append(E_error)
        
        print(f"{T:8.4f} {beta:10.4f} {tau:12.6f} {E_per_site:12.6f} {E_error:10.6f}")
    
    print(f"{'='*70}\n")
    
    # Save results
    if output_file:
        data = {
            'model': model_type,
            'params': params,
            'N': N,
            'temperatures': temps.tolist(),
            'betas': betas.tolist(),
            'taus': taus.tolist(),
            'energies_per_site': [float(e) for e in energies],
            'errors': [float(e) for e in errors]
        }
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {output_file}")
    
    return np.array(temps), np.array(energies), np.array(errors)


# ---------------------------------------------------------------------------
# Example: Temperature sweep on 1D TFIM
# ---------------------------------------------------------------------------

L_chain = 10
g_chain = nk.graph.Chain(length=L_chain, pbc=True)

temps, energies, errors = temperature_sweep(
    graph=g_chain,
    model_type="tfim",
    J=1.0,
    params={"hx": 0.5},
    T_min=0.1,
    T_max=10,
    n_temps=8,
    n_samples=2**16,  # Increased samples for better gradients
    n_chains=16,      # More chains for better statistics
    dt=0.001,        # Smaller timestep for better convergence
    output_file="tfim_temperature_sweep.json"
)


# ---------------------------------------------------------------------------
# Plot results
# ---------------------------------------------------------------------------

# Create plot
plt.figure(figsize=(8, 6))
plt.errorbar(temps, energies, yerr=errors, fmt='o-', capsize=5, 
             markersize=8, linewidth=2, label='NQS TDVP')
plt.xscale('log')
plt.ylim(-1.2, 0)
plt.xticks([0.1, 1, 10], ['0.1', '1', '10'], fontsize= 12)
plt.yticks(fontsize=12)
plt.xlabel('Temperature T', fontsize=14)
plt.ylabel('Energy per site E/N', fontsize=14)
plt.title(f'1D {model} (L={N}, J=1.0, hx={params.get("hx", "N/A")})', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('tfim_temperature_sweep.png', dpi=150, bbox_inches='tight')
print("Plot saved to tfim_temperature_sweep.png")
plt.show()
