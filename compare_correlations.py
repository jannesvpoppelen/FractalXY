import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from pp import PAGE, _corr_panel, _fit, _load_correlators, _save


def files(root, shape, G, seeds, suffix):
    paths = [f"{root}/{shape}/vit_{shape}_G={G}_seed={s}{suffix}.json" for s in seeds]
    return [p for p in paths if os.path.exists(p)]


def plain(label):
    """Legend label without mathtext, for printing"""
    return label.replace("$", "").replace("_{3v}", "3v")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shape", default="triangular")
    p.add_argument("--gen", type=int, default=3)
    p.add_argument("--seeds", type=int, nargs="+", default=[1], help="averaged over, as in pp.py")
    p.add_argument("--sz-root", default="data/ViT_sz")
    args = p.parse_args()
    shape, G = args.shape, args.gen

    new_symm = files(args.sz_root, shape, G, args.seeds, "_symm_observables")
    groups = {
        "original": files("data/ViT", shape, G, args.seeds, "_observables"),
        r"original, C$_{3v}$": files("data/ViT", shape, G, args.seeds, "_symm_observables"),
        r"$S^z$ sector, C$_{3v}$" if new_symm else r"$S^z$ sector":
            new_symm or files(args.sz_root, shape, G, args.seeds, "_observables"),
    }
    print(f"{shape} G={G}, seeds {args.seeds}")
    for label, paths in groups.items():
        if not paths:
            print(f"  {plain(label)}: no observables files")
            continue
        E = [json.load(open(f))["E"] for f in paths]
        print(f"  {plain(label)}: E = {np.mean([e['mean'] for e in E]):.5f} "
              f"+- {np.sqrt(np.sum([e['error'] ** 2 for e in E])) / len(E):.5f}  ({len(paths)} seeds)")
    groups = {label: paths for label, paths in groups.items() if paths}
    graph, eucl, corner = _load_correlators(list(groups.values()), list(groups))

    # retrained against the original projected state, per graph distance
    by_label = {s[3]: s for s in graph}
    old, new = by_label.get(r"original, C$_{3v}$"), graph[-1] if len(groups) > 1 else None
    if old is not None and new is not None and new is not old:
        print(f"\n  C_xy(d): {plain(old[3])} vs {plain(new[3])}")
        for k, d in enumerate(old[0]):
            j = np.flatnonzero(new[0] == d)
            if j.size:
                c0, e0, c1, e1 = old[1][k], old[2][k], new[1][j[0]], new[2][j[0]]
                print(f"    d={d}: {c0:.5f} +- {e0:.5f}   {c1:.5f} +- {e1:.5f}   "
                      f"diff {c1 - c0:+.5f} ({(c1 - c0) / np.hypot(e0, e1):+.1f} sigma)")

    print("\n  power-law fits C = A x^-eta")
    for kind, sets in (("graph", graph), ("euclidean", eucl), ("boundary", corner)):
        for d, c, _, label in sets:
            eta, A = _fit(d, c)
            print(f"    {kind:9s} {plain(label):22s} " + (f"eta = {eta:.4f}, A = {A:.4f}" if eta is not None else "no fit"))

    fig, ax = plt.subplots(1, 2, figsize=(PAGE, 0.42 * PAGE))
    _corr_panel(ax[0], graph, [], r"$d$", "(a)", legend=True, ncol=1)
    _corr_panel(ax[1], eucl, [], r"$r$", "(b)")
    ax[1].set_ylabel("")
    fig.tight_layout(pad=0.15)
    out = f"correlations_{shape}_G={G}_old_vs_sz"
    _save(fig, out)
    print(f"\nsaved img/{out}.pdf")


if __name__ == "__main__":
    main()
