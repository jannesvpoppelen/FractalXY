"""RBM benchmark: gasket vs. clean (hole-free) lattice of comparable size.

Trains one RBM, computes the observables of expect.build_observables and, for
N <= 16, the ED ground-state energy. Results go to data/RBM_bench/.

    python run_rbm_bench.py --shape triangular --lattice clean --size 9 --seed 1
    python run_rbm_bench.py --shape honeycomb --lattice gasket --size 3 --seed 1
"""

import argparse
import json
import os
import time

import numpy as np
import netket as nk

from fractals import (
    triangular_gasket, honeycomb_gasket, honeycomb_gasket_coordinates,
    triangular_lattice, honeycomb_lattice,
)
from utils import build_hamiltonian
from trainRBM import run_xy_model
from expect import build_observables, compute_obs, save_obs

DATA_DIR = "data/RBM_bench"
ED_MAX_SITES = 16


def build_geometry(shape, lattice, size, n=2):
    """size is the generation G for a gasket, the linear size for a clean lattice."""
    if lattice == "gasket":
        if shape == "triangular":
            A, _ = triangular_gasket(size, n)
            x, y, _, _ = honeycomb_gasket_coordinates(size, n)
        else:
            A, _ = honeycomb_gasket(size, n)
            xa, ya, xb, yb = honeycomb_gasket_coordinates(size, n)
            x, y = np.hstack((xa, xb)), np.hstack((ya, yb))
        tag = f"{shape}_gasket_G={size}"
    elif lattice == "clean":
        A, x, y = triangular_lattice(size) if shape == "triangular" else honeycomb_lattice(size)
        tag = f"{shape}_clean_L={size}"
    else:
        raise ValueError(f"lattice must be 'gasket' or 'clean', got {lattice!r}")

    A = A.tocoo()
    g = nk.graph.Graph([(int(i), int(j)) for i, j in zip(A.row, A.col) if i < j])
    positions = np.column_stack((np.asarray(x, float), np.asarray(y, float)))
    return g, positions, tag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", choices=["triangular", "honeycomb"], required=True)
    p.add_argument("--lattice", choices=["gasket", "clean"], required=True)
    p.add_argument("--size", type=int, required=True, help="gasket generation G, or clean linear size")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--alpha", type=int, default=1)
    p.add_argument("--n-samples", type=int, default=2**16)
    p.add_argument("--n-chains", type=int, default=64)
    p.add_argument("--obs-samples", type=int, default=2**14)
    args = p.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    g, positions, tag = build_geometry(args.shape, args.lattice, args.size)
    N = g.n_nodes
    n_bonds = len(list(g.edges()))
    name = f"rbm_{tag}_seed={args.seed}"
    outfile = f"{DATA_DIR}/{name}.mpack"
    print(f"{name}: N={N} sites, {n_bonds} bonds")

    hi = nk.hilbert.Spin(s=0.5, N=N)
    H = build_hamiltonian(hi, g, J=1.0, Jz=0.0, hx=0.0, hy=0.0)

    e_ed = None
    if N <= ED_MAX_SITES:
        t0 = time.time()
        e_ed = float(np.real(np.asarray(nk.exact.lanczos_ed(H, compute_eigenvectors=False, k=1))[0]))
        print(f"  ED ground-state energy: {e_ed:.6f}  ({time.time() - t0:.1f}s)")

    vstate = run_xy_model(
        g, name,
        J=1.0, Jz=0.0, hx=0.0, hy=0.0,
        steps=args.steps, alpha=args.alpha, seed=args.seed,
        n_samples=args.n_samples, n_chains=args.n_chains,
        outfile=outfile, compute_obs=False,
    )

    obs_dict = build_observables(hi, g, positions=positions, H=H)
    vstate.n_samples = args.obs_samples
    results = compute_obs(vstate, obs_dict)
    save_obs(results, f"{DATA_DIR}/{name}_observables.json")

    e_vmc = float(np.real(results["E"].mean))
    summary = {
        "name": name,
        "shape": args.shape,
        "lattice": args.lattice,
        "size": args.size,
        "seed": args.seed,
        "N": int(N),
        "n_bonds": int(n_bonds),
        "E_vmc": e_vmc,
        "E_vmc_err": float(np.real(results["E"].error_of_mean)),
        "E_vmc_per_site": e_vmc / N,
        "E_ed": e_ed,
        "E_ed_per_site": (e_ed / N) if e_ed is not None else None,
        "rel_error": (abs(e_vmc - e_ed) / abs(e_ed)) if e_ed not in (None, 0.0) else None,
        "abs_error_per_site": (abs(e_vmc - e_ed) / N) if e_ed is not None else None,
    }
    with open(f"{DATA_DIR}/{name}_benchmark.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {DATA_DIR}/{name}_benchmark.json")
    if e_ed is not None:
        print(f"  E_vmc/N = {e_vmc / N:.6f}, E_ed/N = {e_ed / N:.6f}, rel. err = {summary['rel_error']:.2e}")


if __name__ == "__main__":
    main()
