import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from fractals import honeycomb_gasket, triangular_gasket

N_PATCH = 2
MAX_STATES = 6 * 10**7   # refuse sectors whose matrix would not fit in memory


def target_sz(N):
    return 0.0 if N % 2 == 0 else 0.5


def bonds(shape, G):
    build = triangular_gasket if shape == "triangular" else honeycomb_gasket
    A = build(G, N_PATCH)[0].tocoo()
    return A.shape[0], sorted({(int(min(i, j)), int(max(i, j))) for i, j in zip(A.row, A.col) if i != j})


def sector_states(N, n_up):
    """Sorted integers whose binary representation has n_up ones (bit i = site i up)."""
    if N > 34:
        raise ValueError(f"N={N} is too large for exact diagonalisation")
    states = np.arange(2**N, dtype=np.int64)
    count = np.zeros(states.shape, dtype=np.int8)
    for i in range(N):
        count += ((states >> i) & 1).astype(np.int8)
    return states[count == n_up]


def sector_hamiltonian(N, edges, total_sz, J=1.0):
    n_up = int(round(N / 2 + total_sz))
    states = sector_states(N, n_up)
    dim = len(states)
    rows, cols = [], []
    for i, j in edges:
        differ = ((states >> i) & 1) != ((states >> j) & 1)
        src = np.nonzero(differ)[0]
        target = states[src] ^ ((1 << i) | (1 << j))
        rows.append(src)
        cols.append(np.searchsorted(states, target))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    H = sp.csr_matrix((np.full(len(rows), -2.0 * J), (rows, cols)), shape=(dim, dim))
    return H, dim


def lowest(H, k):
    if H.shape[0] <= 2000:
        return np.linalg.eigvalsh(H.toarray())[:k]
    return np.sort(sla.eigsh(H, k=k, which="SA", tol=1e-12)[0])


def compare_with_seeds(shape, G, e0):
    """Energies of the trained seeds, from sector_fidelity.py output, against the exact value."""
    for root in ("data/ViT", "data/ViT_sz"):
        for tag in ("_symm", ""):
            path = f"{root}/{shape}/fidelity_{shape}_gasket_G={G}_sector{tag}.json"
            if not os.path.exists(path):
                continue
            d = json.load(open(path))
            print(f"  seeds in {path}:")
            for seed, e in d["energies"].items():
                diff = e["mean"] - e0
                sigma = f"{diff / e['error']:+.1f} sigma" if e["error"] > 0 else "exact"
                print(f"    seed {seed}: E = {e['mean']:.6f}  E - E_ED = {diff:+.2e}  ({sigma})")
            break


def run(shape, G, total_sz, k):
    N, edges = bonds(shape, G)
    s0 = target_sz(N) if total_sz is None else total_sz
    from math import comb
    dim = comb(N, int(round(N / 2 + s0)))
    print(f"\n{shape} G={G}: N={N}, {len(edges)} bonds, S^z={s0:+}, {dim} states")
    if dim > MAX_STATES:
        print(f"  skipped: more than {MAX_STATES} states")
        return

    t0 = time.time()
    H, dim = sector_hamiltonian(N, edges, s0)
    t1 = time.time()
    print(f"  matrix built in {t1 - t0:.1f} s ({H.nnz} nonzeros)")
    E = lowest(H, k)
    t2 = time.time()
    print(f"  lowest energies {np.array2string(E, precision=10)}  ({t2 - t1:.1f} s)")
    print(f"  E0 = {E[0]:.10f}, E0/N = {E[0] / N:.10f}, gap to next level in the sector {E[1] - E[0]:.6f}")

    os.makedirs("data/ED", exist_ok=True)
    out = f"data/ED/{shape}_G={G}_sz={s0:+g}.json"
    with open(out, "w") as f:
        json.dump({"shape": shape, "generation": G, "n_sites": N, "sector_sz": s0, "dim": dim,
                   "n_bonds": len(edges), "J": 1.0, "Jz": 0.0, "energies": E.tolist(),
                   "time_build_s": t1 - t0, "time_lanczos_s": t2 - t1}, f, indent=2)
    print(f"  saved {out}")
    compare_with_seeds(shape, G, E[0])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shapes", nargs="+", default=["triangular", "honeycomb"])
    p.add_argument("--gens", type=int, nargs="+", default=[1, 2])
    p.add_argument("--sz", type=float, default=None, help="sector; default 0 (even N) or +1/2 (odd N)")
    p.add_argument("--k", type=int, default=2, help="number of lowest levels")
    args = p.parse_args()

    for shape in args.shapes:
        for G in args.gens:
            run(shape, G, args.sz, args.k)
