"""XY model simulation on fractal and regular lattices using NetKet."""

import json
import os
import time
from collections import defaultdict

import numpy as np

import jax
import jax.random as jr
import netket as nk

from fractals import honeycomb_gasket, triangular_gasket
from utils import build_hamiltonian, build_magnetizations, save_vstate, save_metadata

print(f"NetKet {nk.__version__}, Jax {jax.__version__}")
print(f"JAX is using: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")


def normalize_feature_dims(feature_dims, num_layers):
    """Return a tuple of feature sizes with one entry per layer."""
    if feature_dims is None:
        return tuple([8] * num_layers)

    if isinstance(feature_dims, int):
        return tuple([feature_dims] * num_layers)

    dims = tuple(feature_dims)
    if len(dims) != num_layers:
        raise ValueError(
            f"feature_dims has length {len(dims)} but num_layers={num_layers}. "
            "Pass an int or a tuple/list with one value per layer."
        )
    return dims


def run_xy_model(
    g,
    name,
    J=1.0,
    Jz=0.0,
    hx=None,
    hy=None,
    steps=250,
    seed=None,
    n_samples=2**16,
    n_chains=32,
    outfile="vstate.mpack",
    compute_obs=False,
    num_layers=6,
    feature_dims=None,
):
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
        E, _ = nk.exact.lanczos_ed(H, compute_eigenvectors=True, k=3)
        E_0 = E[0]
        E_1 = E[1]
        E_2 = E[2]
        gap1 = E_1 - E_0
        gap2 = E_2 - E_0
        print(f"{name} ED gaps: E1-E0={gap1}, E2-E0={gap2}")
        print(f"{name} ED ground state: {E_0}")

    obs_dict = build_magnetizations(hi, g) if compute_obs else None

    symmetries = g.automorphisms()
    feature_dims = normalize_feature_dims(feature_dims, num_layers)
    model = nk.models.GCNN(symmetries=symmetries, layers=num_layers, features=feature_dims)
    # model = nk.models.RBM(alpha=1)

    sampler = nk.sampler.MetropolisLocal(hi, n_chains=n_chains)
    n_discard_per_chain = 2**7
    vstate = nk.vqs.MCState(
        sampler,
        model,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        seed=seed,
        chunk_size=None,
    )

    lr = 0.01
    optimizer_name = "Adam"  # Set manually if you change optimizer.
    optimizer = nk.optimizer.Adam(learning_rate=lr)
    print("NQS parameters: ", vstate.n_parameters)

    gs = nk.driver.VMC(hamiltonian=H, optimizer=optimizer, variational_state=vstate)
    log_file = f"{name}" if compute_obs else None
    gs.run(n_iter=steps, obs=obs_dict, out=log_file)

    E = vstate.expect(H)
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

    # Save metadata
    metadata_file = outfile.replace(".mpack", "_metadata.json")
    model_config = {
        "type": "GCNN",
        "num_layers": num_layers,
        "feature_dims": list(feature_dims),
        "kernel_init": "normal(stddev=0.01)",
        "hidden_bias_init": "normal(stddev=0.01)",
    }
    sampler_config = {
        "type": "MetropolisLocal",
        "n_chains": n_chains,
        "n_samples": n_samples,
        "n_discard_per_chain": n_discard_per_chain,
        "seed": seed,
    }
    hamiltonian_config = {
        "J": J,
        "Jz": Jz,
        "hx": float(hx),
        "hy": float(hy),
    }
    optimizer_config = {
        "type": optimizer_name,
        "learning_rate": lr,
    }
    # Store final energy and variance
    final_results = {
        "final_energy": float(np.real(E.mean)),
        "final_variance": float(np.real(Evar)),
        "training_time": end - start,
        "n_parameters": vstate.n_parameters,
    }
    save_metadata(
        metadata_file,
        model_config,
        sampler_config,
        hamiltonian_config,
        optimizer_config=optimizer_config,
        final_results=final_results,
    )
    print(f"Saved metadata to {metadata_file}")

    return vstate


def run_gcnn_hyperparameter_search(
    g,
    base_name,
    layer_candidates,
    feature_candidates,
    seeds,
    run_kwargs=None,
    results_file="gcnn_hparam_search.json",
):
    """Grid search over GCNN layers and feature dimensions.

    Args:
        g: NetKet graph.
        base_name: Prefix used for trial output files.
        layer_candidates: Iterable of candidate layer counts.
        feature_candidates: Iterable of ints or tuples/lists. If int, same width for all layers.
        seeds: Iterable of random seeds.
        run_kwargs: Extra kwargs forwarded to run_xy_model.
        results_file: JSON output with all trials and aggregated stats.
    """
    run_kwargs = dict(run_kwargs or {})
    trials = []

    total_trials = len(layer_candidates) * len(feature_candidates) * len(seeds)
    trial_idx = 0

    for num_layers in layer_candidates:
        for feature_spec in feature_candidates:
            try:
                feature_dims = normalize_feature_dims(feature_spec, num_layers)
            except ValueError as err:
                print(
                    f"Skipping layers={num_layers}, feature_spec={feature_spec}: {err}"
                )
                continue
            feature_tag = "-".join(str(x) for x in feature_dims)

            for seed in seeds:
                trial_idx += 1
                trial_name = f"{base_name}_L{num_layers}_F{feature_tag}_seed{seed}"
                outfile = f"{trial_name}.mpack"

                print(f"\n[{trial_idx}/{total_trials}] Running {trial_name}")
                run_xy_model(
                    g,
                    trial_name,
                    seed=seed,
                    outfile=outfile,
                    num_layers=num_layers,
                    feature_dims=feature_dims,
                    **run_kwargs,
                )

                metadata_file = outfile.replace(".mpack", "_metadata.json")
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                result = {
                    "trial_name": trial_name,
                    "seed": seed,
                    "num_layers": num_layers,
                    "feature_dims": list(feature_dims),
                    "final_energy": metadata["final_results"]["final_energy"],
                    "final_variance": metadata["final_results"]["final_variance"],
                    "training_time": metadata["final_results"]["training_time"],
                    "n_parameters": metadata["final_results"]["n_parameters"],
                    "vstate_file": outfile,
                    "metadata_file": metadata_file,
                    "vstate_exists": os.path.exists(outfile),
                    "metadata_exists": os.path.exists(metadata_file),
                }
                trials.append(result)

    grouped = defaultdict(list)
    for t in trials:
        key = (t["num_layers"], tuple(t["feature_dims"]))
        grouped[key].append(t)

    aggregated = []
    for (num_layers, feature_dims), group_trials in grouped.items():
        energies = np.array([x["final_energy"] for x in group_trials], dtype=float)
        variances = np.array([x["final_variance"] for x in group_trials], dtype=float)
        times = np.array([x["training_time"] for x in group_trials], dtype=float)

        aggregated.append(
            {
                "num_layers": num_layers,
                "feature_dims": list(feature_dims),
                "n_trials": len(group_trials),
                "mean_energy": float(np.mean(energies)),
                "std_energy": float(np.std(energies)),
                "mean_variance": float(np.mean(variances)),
                "mean_training_time": float(np.mean(times)),
            }
        )

    aggregated.sort(key=lambda x: x["mean_energy"])

    output = {
        "search_space": {
            "layer_candidates": list(layer_candidates),
            "feature_candidates": [
                int(x) if isinstance(x, int) else list(x) for x in feature_candidates
            ],
            "seeds": list(seeds),
        },
        "n_total_trials": len(trials),
        "best_config_by_mean_energy": aggregated[0] if aggregated else None,
        "aggregated": aggregated,
        "trials": trials,
    }

    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved hyperparameter search results to {results_file}")
    if aggregated:
        best = aggregated[0]
        print(
            "Best config (mean energy): "
            f"layers={best['num_layers']}, "
            f"features={best['feature_dims']}, "
            f"mean_E={best['mean_energy']:.6f} +/- {best['std_energy']:.6f}"
        )

    return output


if __name__ == "__main__":
    generations = [2]
    seeds = [1]
    shapes = ["hexagonal"]
    n = 2
    for shape in shapes:
        for G in generations:
            print(f"\n{'=' * 60}")
            print(f"Running generation {G}")
            print(f"{'=' * 60}\n")

            if shape == "triangular":
                A, _ = triangular_gasket(G, n)
            else:
                A, _ = honeycomb_gasket(G, n)

            A = A.tocoo()
            edges = list(zip(A.row, A.col))
            edges_list = [(i, j) for i, j in edges if i < j]

            g_sierpinski = nk.graph.Graph(edges_list)
            print(f"{shape} gasket G={G}: {g_sierpinski.n_nodes} nodes\n")

            layer_candidates = [5]
            feature_candidates = [12]
            run_kwargs = {
                "steps": 500,
                "compute_obs": False,
                "n_samples": 2**4,
                "n_chains": 32,
                "hy": 0,
                "hx": 0,
            }

            run_gcnn_hyperparameter_search(
                g_sierpinski,
                base_name=f"data/GCNN/{shape}_gasket_G={G}",
                layer_candidates=layer_candidates,
                feature_candidates=feature_candidates,
                seeds=seeds,
                run_kwargs=run_kwargs,
                results_file=f"{shape}_gasket_G={G}_gcnn_hparam_search.json",
            )

