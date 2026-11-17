import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from compare_correlations import files, plain
from pp import COLORS, MARKERS, PAGE, _load_correlators, _pow10, _save, _tag


def weighted_fit(d, c, err, dmin=None, dmax=None):
    """Fit log C = log A - eta log d with weights C / err.

    Returns eta, delta eta, A, chi^2/dof and the mask of the fitted points; eta is None for
    fewer than 2 points, delta eta and chi^2/dof are nan for exactly 2.
    """
    m = (c > 0) & (err > 0)
    if dmin is not None:
        m &= d >= dmin
    if dmax is not None:
        m &= d <= dmax
    n = m.sum()
    if n < 2:
        return None, np.nan, None, np.nan, m
    x, y, sy = np.log(d[m]), np.log(c[m]), err[m] / c[m]
    if n == 2:
        slope, intercept = np.polyfit(x, y, 1)
        return -slope, np.nan, np.exp(intercept), np.nan, m
    (slope, intercept), cov = np.polyfit(x, y, 1, w=1 / sy, cov="unscaled")
    chi2 = np.sum(((y - (slope * x + intercept)) / sy) ** 2) / (n - 2)
    deta = np.sqrt(cov[0, 0] * max(1.0, chi2))
    return -slope, deta, np.exp(intercept), chi2, m


def fmt_eta(eta, deta):
    if eta is None:
        return "no fit"
    return rf"$\eta={eta:.3f}$" if np.isnan(deta) else rf"$\eta={eta:.3f}\pm{deta:.3f}$"


def panel(ax, series, xlabel, tag, dmin, dmax):
    for i, (d, c, err, label) in enumerate(series):
        col, mk = COLORS[i % len(COLORS)], MARKERS[i % len(MARKERS)]
        eta, deta, A, _, m = weighted_fit(d, c, err, dmin, dmax)
        pos = c > 0
        ax.errorbar(d[m], c[m], yerr=err[m], fmt=mk, color=col, ms=4.5, lw=0.8, capsize=2,
                    ls="none", zorder=2, label=f"{label}: {fmt_eta(eta, deta)}")
        out = pos & ~m
        ax.errorbar(d[out], c[out], yerr=err[out], fmt=mk, color=col, mfc="none", ms=4.5,
                    lw=0.8, capsize=2, ls="none", zorder=2, alpha=0.6)
        if eta is not None:
            r = np.logspace(np.log10(d[m].min()), np.log10(d[m].max()), 200)
            ax.plot(r, A * r ** -eta, color=col, lw=1.2, ls="--", zorder=3)

    ax.set(xscale="log", yscale="log")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_pow10))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    lo, hi = min(s[1][s[1] > 0].min() for s in series), max(s[1].max() for s in series)
    if hi / lo < 10:  # narrow range: a few plain ticks instead of crowded log labels
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=11)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1,
              edgecolor="0.5", labelspacing=0.2, handlelength=1.3)
    _tag(ax, tag, x=0.97, y=0.97)


def run(shape, G, seeds, sz_root, dmin, dmax):
    new_symm = files(sz_root, shape, G, seeds, "_symm_observables")
    groups = {
        "original": files("data/ViT", shape, G, seeds, "_observables"),
        r"original, C$_{3v}$": files("data/ViT", shape, G, seeds, "_symm_observables"),
        r"$S^z$ sector, C$_{3v}$" if new_symm else r"$S^z$ sector":
            new_symm or files(sz_root, shape, G, seeds, "_observables"),
        r"original, C$_{3v}$, $S^z$ projected": files("data/ViT", shape, G, seeds, "_sector_observables"),
    }
    groups = {label: paths for label, paths in groups.items() if paths}
    if not groups:
        print(f"{shape} G={G}: no observables files")
        return
    graph, eucl, corner = _load_correlators(list(groups.values()), list(groups))

    print(f"\n{shape} G={G}, seeds {seeds}, window [{dmin}, {dmax}]")
    for label, paths in groups.items():
        print(f"  {plain(label)}: {len(paths)} seeds")
    print(f"  {'distance':9s} {'state':22s} {'eta':>8s} {'d_eta':>8s} {'A':>8s} {'chi2/dof':>9s} {'n':>3s}")
    for kind, sets in (("graph", graph), ("euclidean", eucl), ("boundary", corner)):
        for d, c, err, label in sets:
            eta, deta, A, chi2, m = weighted_fit(d, c, err, dmin, dmax)
            if eta is None:
                print(f"  {kind:9s} {plain(label):22s} no fit")
            else:
                print(f"  {kind:9s} {plain(label):22s} {eta:8.4f} {deta:8.4f} {A:8.4f} {chi2:9.2f} {m.sum():3d}")

    panels = [(s, xl, t) for s, xl, t in ((graph, r"$d$", "(a)"), (eucl, r"$r$", "(b)"),
                                         (corner, r"$d_\partial$", "(c)")) if s]
    fig, ax = plt.subplots(1, len(panels), figsize=(PAGE, 0.5 * PAGE), squeeze=False)
    for a, (s, xl, t) in zip(ax[0], panels):
        panel(a, s, xl, t, dmin, dmax)
    ax[0, 0].set_ylabel(r"$C_{xy}$", fontsize=14)
    fig.tight_layout(pad=0.15)
    out = f"fits_{shape}_G={G}"
    _save(fig, out)
    print(f"  saved img/{out}.pdf")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shape", default="triangular")
    p.add_argument("--gens", type=int, nargs="+", default=[3])
    p.add_argument("--seeds", type=int, nargs="+", default=[1], help="averaged over, as in pp.py")
    p.add_argument("--sz-root", default="data/ViT_sz")
    p.add_argument("--dmin", type=float, default=None, help="smallest distance in the fit")
    p.add_argument("--dmax", type=float, default=None, help="largest distance in the fit")
    args = p.parse_args()
    for G in args.gens:
        run(args.shape, G, args.seeds, args.sz_root, args.dmin, args.dmax)


if __name__ == "__main__":
    main()
