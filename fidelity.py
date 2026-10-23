import gc
import json

import numpy as np
import jax
import netket as nk
from netket.experimental.observable import InfidelityOperator

from trainViT import build_fractal, patch_distances
from expect import load_vstate_vit, load_vstate_vit_symm

print(f"NetKet {nk.__version__}, Jax {jax.__version__}")
print(f"JAX is using: {jax.devices()}")

SKIP_SYMM_GENS = {5}  # no projected states saved for these


def compute_fidelity(vstate1, vstate2):
    # cv_coeff=None: the default control-variate estimator is unbounded below and
    # blows up when psi_1/psi_2 has heavy tails on the samples
    obs = InfidelityOperator(vstate1, cv_coeff=None)
    vstate2.n_samples = 2**13
    result = vstate2.expect(obs)
    fidelity = 1.0 - complex(result.mean.item())
    if abs(fidelity.imag) > 1e-2:
        print(f"  warning: Im F = {fidelity.imag:.3g}")
    return fidelity.real


def compute_pairwise_fidelities(vstates, seeds):
    fidelities = {}
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            fid = compute_fidelity(vstates[i], vstates[j])
            fidelities[f"seed{seeds[i]}_seed{seeds[j]}"] = float(fid)
            print(f"  F(seed {seeds[i]}, seed {seeds[j]}) = {fid}")
    return fidelities


def summarize_fidelities(shape, G, seeds, fidelities):
    values = list(fidelities.values())
    return {
        "shape": shape,
        "generation": G,
        "seeds": seeds,
        "fidelities": fidelities,
        "mean_fidelity": float(np.mean(values)),
        "std_fidelity": float(np.std(values)),
        "min_fidelity": float(np.min(values)),
        "max_fidelity": float(np.max(values)),
    }


def run(shape, G, seeds, hi, g, patches, distances, symm):
    vstates = []
    for seed in seeds:
        base = f"data/ViT/{shape}/vit_{shape}_G={G}_seed={seed}"
        if symm:
            vstates.append(load_vstate_vit_symm(f"{base}_symm.mpack", f"{base}_metadata.json", hi, g, patches, distances))
        else:
            vstates.append(load_vstate_vit(f"{base}.mpack", f"{base}_metadata.json", hi, patches, distances))

    print("After symmetry projection:" if symm else "Before symmetry projection:")
    results = summarize_fidelities(shape, G, seeds, compute_pairwise_fidelities(vstates, seeds))
    print(f"  mean F = {results['mean_fidelity']:.4f} ± {results['std_fidelity']:.4f}")

    filename = f"data/ViT/{shape}/fidelity_{shape}_gasket_G={G}{'_symm' if symm else ''}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {filename}")

    del vstates
    gc.collect()
    jax.clear_caches()


if __name__ == "__main__":
    gens = [1, 2, 3, 4, 5]
    seeds = [1, 2, 3, 4, 5]
    shapes = ["triangular", "honeycomb"]
    n = 2

    for shape in shapes:
        for G in gens:
            g, patches, x, y = build_fractal(shape, G, n)
            distances = patch_distances(patches, x, y)
            hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
            print(f"\n{shape} gasket G={G}: {g.n_nodes} nodes")

            run(shape, G, seeds, hi, g, patches, distances, symm=False)
            if G not in SKIP_SYMM_GENS:
                run(shape, G, seeds, hi, g, patches, distances, symm=True)
