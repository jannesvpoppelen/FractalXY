"""RBM ansatz for the XY model on fractal lattices."""

import os
import time

import numpy as np
import jax
import jax.random as jr
import netket as nk
from flax import linen as nn

from utils import build_hamiltonian, build_magnetizations, save_vstate, save_metadata
from fractals import triangular_gasket, honeycomb_gasket

print(f"NetKet {nk.__version__}, Jax {jax.__version__}")
print(f"JAX is using: {jax.devices()}")


def run_xy_model(g, name, J=1.0, Jz=0.0, hx=None, hy=None, steps=250, alpha=1, seed=None, n_samples=2**16, n_chains=32, outfile="vstate.mpack", compute_obs=False):
    start = time.time()

    # small random field to break the U(1) degeneracy, unless hx/hy are given
    if hx is None or hy is None:
        key1, key2 = jr.split(jr.PRNGKey(seed))
        if hx is None:
            hx = 0.01 * jr.uniform(key1, minval=-1.0, maxval=1.0)
        if hy is None:
            hy = 0.01 * jr.uniform(key2, minval=-1.0, maxval=1.0)

    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
    H = build_hamiltonian(hi, g, J, Jz, hx, hy)

    if g.n_nodes < 16:
        E, _ = nk.exact.lanczos_ed(H, compute_eigenvectors=True, k=3)
        print(f"{name} ED gaps: E1-E0={E[1] - E[0]}, E2-E0={E[2] - E[0]}")
        print(f"{name} ED ground state: {E[0]}")

    obs_dict = build_magnetizations(hi, g) if compute_obs else None

    model = nk.models.RBM(alpha=alpha, kernel_init=nn.initializers.normal(stddev=0.01), hidden_bias_init=nn.initializers.normal(stddev=0.01))
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=n_chains)
    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain=2**7, seed=seed)

    lr = 0.01
    optimizer = nk.optimizer.Sgd(learning_rate=lr)

    gs = nk.driver.VMC_SR(hamiltonian=H, optimizer=optimizer, variational_state=vstate, diag_shift=0.01)
    log_file = f"{name}" if compute_obs else None
    gs.run(n_iter=steps, obs=obs_dict, out=log_file)

    E = vstate.expect(H)
    Evar = E.variance
    print(f"E: {np.real(E.mean)}, σ²: {np.real(Evar)}")

    end = time.time()
    print("NQS parameters: ", vstate.n_parameters)
    print("Training time: ", end - start, " seconds")

    save_vstate(vstate, outfile)

    # the netket logger also writes the vstate; we already saved it
    if compute_obs and os.path.exists(f"{name}.log.mpack"):
        os.remove(f"{name}.log.mpack")

    metadata_file = outfile.replace(".mpack", "_metadata.json")
    model_config = {
        "type": "RBM",
        "alpha": alpha,
        "kernel_init": "normal(stddev=0.01)",
        "hidden_bias_init": "normal(stddev=0.01)",
    }
    sampler_config = {
        "type": "MetropolisLocal",
        "n_chains": n_chains,
        "n_samples": n_samples,
        "n_discard_per_chain": 2**7,
        "seed": seed,
    }
    hamiltonian_config = {"J": J, "Jz": Jz, "hx": float(hx), "hy": float(hy)}
    final_results = {
        "final_energy": float(np.real(E.mean)),
        "final_variance": float(np.real(Evar)),
        "training_time": end - start,
        "n_parameters": vstate.n_parameters,
    }
    save_metadata(metadata_file, model_config, sampler_config, hamiltonian_config, final_results=final_results)
    print(f"Saved metadata to {metadata_file}")

    return vstate


if __name__ == "__main__":
    generations = [1, 2, 3, 4]
    seeds = [1, 2, 3, 4, 5]
    shapes = ["triangular", "honeycomb"]
    n = 2
    for shape in shapes:
        for G in generations:
            for seed in seeds:
                A, _ = triangular_gasket(G, n) if shape == "triangular" else honeycomb_gasket(G, n)
                A = A.tocoo()
                g = nk.graph.Graph([(i, j) for i, j in zip(A.row, A.col) if i < j])

                name = f"{shape}_gasket_G={G}_seed={seed}"
                print(f"\n{shape} gasket G={G}: {g.n_nodes} nodes")
                run_xy_model(g, name, steps=500, alpha=1, seed=seed, outfile=f"{name}.mpack", compute_obs=True, n_samples=2**16, n_chains=32, hy=0, hx=0)
