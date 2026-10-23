"""One point of the ViT hyperparameter grid search.

    python hyperSearchViT.py --num_layers 4 --d_model 32 --n_heads 8
"""

import argparse
import json
import os

import numpy as np
import jax
import netket as nk

from trainViT import build_fractal, run_xy_model

FIXED = dict(
    J=1.0,
    Jz=0.0,
    hx=0.0,
    hy=0.0,
    N_samples=2**12,
    N_opt=750,
    N_symm_opt=250,
    n_chains=2**12,
    early_stop_patience=150,
)

SHAPE = "honeycomb"
G = 4
N = 2
SEED = 1
OUTDIR = "data/ViT_hypersearch"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num_layers", type=int, required=True)
    p.add_argument("--d_model", type=int, required=True)
    p.add_argument("--n_heads", type=int, required=True)
    args = p.parse_args()
    num_layers, d_model, n_heads = args.num_layers, args.d_model, args.n_heads
    assert d_model % n_heads == 0, f"n_heads={n_heads} does not divide d_model={d_model}"

    print(f"NetKet {nk.__version__}, JAX {jax.__version__}, devices {jax.devices()}")
    print(f"num_layers={num_layers}, d_model={d_model}, n_heads={n_heads}")

    os.makedirs(OUTDIR, exist_ok=True)
    g, patches, x, y = build_fractal(SHAPE, G, N)
    print(f"{SHAPE} gasket G={G}: {g.n_nodes} nodes, {len(patches)} patches")

    name = f"vit_{SHAPE}_G={G}_seed={SEED}_nl{num_layers}_dm{d_model}_nh{n_heads}"
    base = os.path.join(OUTDIR, name)

    result = run_xy_model(
        g=g,
        name=name,
        patches=patches,
        x=x,
        y=y,
        num_layers=num_layers,
        d_model=d_model,
        n_heads=n_heads,
        seed=SEED,
        outfile=base + ".mpack",
        fractal_config={"shape": SHAPE, "G": G, "n": N},
        **FIXED,
    )

    record = {
        "shape": SHAPE,
        "G": G,
        "seed": SEED,
        "num_layers": num_layers,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_params": int(result["n_params"]),
        "final_energy": result["final_energy"],
        "final_variance": result["final_variance"],
        "stopped_at_iter": int(result["stopped_at_iter"]),
        "training_time": result["training_time"],
    }
    if len(result["symm_energies"]) > 0:
        record["symm_final_energy"] = float(np.mean(result["symm_energies"][-50:]))
        record["symm_final_variance"] = float(np.mean(result["symm_variances"][-50:]))

    with open(base + "_result.json", "w") as f:
        json.dump(record, f, indent=2)
    print(f"final_energy={record['final_energy']:.6f}, n_params={record['n_params']:,}")
