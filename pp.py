"""Figures for the paper. `python pp.py` rebuilds fig1-fig6 from data/.
Needs scienceplots, and pdftoppm for fig2."""

import json
import subprocess
import tempfile
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib import patheffects, ticker
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from PIL import Image, ImageChops
import scienceplots  # noqa: F401

from fractals import honeycomb_gasket, honeycomb_gasket_coordinates, triangular_gasket

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif"})

PAGE = 6.75   # PRB double-column width [inch]
COL = 3.375   # PRB single-column width [inch]
# Okabe-Ito palette
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#999999"]
MARKERS = ["o", "s", "^", "D", "v", "p", "<"]
GREY, GREY_MARK = ["0.15", "0.55"], ["o", "s"]     # clean-lattice reference


def _save(fig, out):
    fig.savefig(f"img/{out}.pdf", dpi=600, bbox_inches="tight")

def _tag(ax, text, x=0.91, y=0.90):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=11, ha="right", va="top",
            zorder=10, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))


def _fit(d, c):
    """Fit C = A d^-eta to the positive points."""
    m = c > 0
    if m.sum() < 2:
        return None, None
    slope, intercept = np.polyfit(np.log(d[m]), np.log(c[m]), 1)
    return -slope, np.exp(intercept)



def _gasket_geometry(shape, g, n=2):
    """Site coordinates and bond segments of the generation-g gasket."""
    xa, ya, xb, yb = honeycomb_gasket_coordinates(g, n)
    if shape == "triangular":
        adj, _ = triangular_gasket(g, n)
        x, y = xa, ya
    else:
        adj, _ = honeycomb_gasket(g, n)
        x, y = np.hstack((xa, xb)), np.hstack((ya, yb))
    a = adj.tocoo()
    m = a.row < a.col
    bonds = np.stack([np.c_[x[a.row[m]], y[a.row[m]]], np.c_[x[a.col[m]], y[a.col[m]]]], axis=1)
    return x, y, bonds


def fig1(gens=(1, 2, 4), n=2, out="fig1"):
    """Triangular (a) and honeycomb (b) gaskets, one column per generation."""
    shapes = ["triangular", "honeycomb"]
    ncol = len(gens)
    w = COL / ncol
    fig, axes = plt.subplots(2, ncol, figsize=(COL, 2 * w * np.sqrt(3) / 2 + 0.30),
                             gridspec_kw=dict(wspace=0.10, hspace=0.22), squeeze=False)
    for r, shape in enumerate(shapes):
        for c, g in enumerate(gens):
            ax = axes[r, c]
            x, y, bonds = _gasket_geometry(shape, g, n)
            x0, y0, side = x.min(), y.min(), x.max() - x.min()
            x, y, bonds = (x - x0) / side, (y - y0) / side, (bonds - [x0, y0]) / side
            bond_pt = 72 * w / 1.08 * np.hypot(*(bonds[0, 1] - bonds[0, 0]))
            lw = np.clip(0.06 * bond_pt, 0.3, 0.9)
            dot = np.clip(0.25 * bond_pt, 1.0, 3.6)
            ax.add_collection(LineCollection(bonds, colors="0.2", linewidths=lw, zorder=1,
                                             capstyle="round"))
            ax.scatter(x, y, s=dot**2, c=COLORS[0], edgecolors="none", zorder=2)
            ax.set(xlim=(-0.04, 1.04), ylim=(-0.04, np.sqrt(3) / 2 + 0.04), aspect="equal")
            ax.axis("off")
            ax.text(0.5, -0.07, rf"$G={g}$", transform=ax.transAxes, ha="center", va="top", fontsize=8)
        axes[r, 0].text(-0.02, 1.0, f"({'ab'[r]})", transform=axes[r, 0].transAxes,
                        ha="left", va="top", fontsize=9)
    _save(fig, out)



def _rasterize(pdf, dpi=600):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile", pdf, f"{tmp}/p"], check=True)
        im = Image.open(f"{tmp}/p.png").convert("RGB")
        bbox = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
        return im.crop(bbox) if bbox else im


def fig2(pdf="vit_fractal.pdf", out="fig2"):
    img = _rasterize(pdf)
    fig = plt.figure(figsize=(PAGE, PAGE * img.height / img.width))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis("off")
    fig.savefig(f"img/{out}.pdf", dpi=600, bbox_inches="tight", pad_inches=0.05)



def _load_history(path):
    d = json.load(open(path))
    return (np.array(d["energies"]), np.array(d["variances"]),
            d.get("symm_start_iter", len(d["energies"])),
            np.array(d.get("symm_energies", [])), np.array(d.get("symm_variances", [])))


def _n_sites(shape, g, n=2):
    build = triangular_gasket if shape == "triangular" else honeycomb_gasket
    return build(g, n)[0].shape[0]


def _conv_panel(ax, files, gens, shape, labels, kind, skip=0, lw=2.2):
    """kind is "energy" or "variance"; dashed = after symmetry projection."""
    for i, (path, g, label) in enumerate(zip(files, gens, labels)):
        if i < skip or path is None:
            continue
        E, V, start, sE, sV = _load_history(path)
        N = _n_sites(shape, g)
        main, tail = (E, sE) if kind == "energy" else (np.clip(V, 1e-12, None), np.clip(sV, 1e-12, None))
        color = COLORS[i % len(COLORS)]
        ax.plot(np.arange(len(main)), main / N, color=color, lw=lw, label=label)
        if len(tail):
            ax.plot(np.arange(start, start + len(tail)), tail / N, color=color, lw=lw, ls="--")
            ax.axvline(start, color=color, lw=0.7, ls="--", alpha=0.35, zorder=0)
    ax.grid(alpha=0.3, ls="--", lw=0.4)
    ax.tick_params(direction="in", top=True, right=True, labelsize=12)
    if kind == "variance":
        ax.set_yscale("log")


def fig3(tri_files, hc_files, gens, labels=None, out="fig3", var_skip=1, var_floor=1e-5):
    """E/N and Var(E)/N vs iteration. var_skip drops the smallest generations
    from the variance row."""
    labels = labels or [rf"$G={g}$" for g in gens]
    fig, ax = plt.subplots(2, 2, figsize=(PAGE, PAGE * 0.64), sharex=True, sharey="row")

    _conv_panel(ax[0, 0], tri_files, gens, "triangular", labels, "energy")
    _conv_panel(ax[0, 1], hc_files, gens, "honeycomb", labels, "energy")
    _conv_panel(ax[1, 0], tri_files, gens, "triangular", labels, "variance", skip=var_skip)
    _conv_panel(ax[1, 1], hc_files, gens, "honeycomb", labels, "variance", skip=var_skip)
    ax[1, 0].set_ylim(bottom=var_floor)
    ax[1, 1].set_ylim(bottom=var_floor)

    hist = {f: _load_history(f) for f in (*tri_files, *hc_files) if f}
    ax[0, 0].set_xlim(0, max((s + len(sE) for _, _, s, sE, _ in hist.values()), default=1))

    ax[0, 0].set_ylabel(r"$E/N$", fontsize=12)
    ax[1, 0].set_ylabel(r"$\mathrm{Var}(E)/N$", fontsize=12)
    ax[1, 0].set_xlabel("Iteration", fontsize=12)
    ax[1, 1].set_xlabel("Iteration", fontsize=12)
    ax[0, 0].legend(fontsize=11, ncol=2, loc="upper left")

    starts = [hist[f][2] for f in tri_files if f]
    if starts:
        ax[0, 0].text(max(starts) - 10, 0.5, "symmetry projection", rotation=90, fontsize=9,
                      color="0.4", ha="right", va="center", transform=ax[0, 0].get_xaxis_transform())

    for a, tag in zip(ax.flat, ["(a)", "(b)", "(c)", "(d)"]):
        _tag(a, tag)

    fig.tight_layout(pad=0.25)
    fig.subplots_adjust(hspace=0.07, wspace=0.12)
    _save(fig, out)



def _avg_over_seeds(seed_obs, prefix, cast):
    keys = sorted({k for o in seed_obs for k in o if k.startswith(prefix)},
                  key=lambda k: cast(k[len(prefix):]))
    if not keys:
        return None
    d = np.array([cast(k[len(prefix):]) for k in keys])
    c = np.array([np.mean([o[k]["mean"] for o in seed_obs if k in o]) for k in keys])
    e = np.array([np.sqrt(np.sum([np.real(o[k]["error"]) ** 2 for o in seed_obs if k in o]))
                  / sum(k in o for o in seed_obs) for k in keys])
    return d, c, e


def _load_correlators(file_groups, labels):
    """file_groups: per series a path or a list of per-seed paths (averaged)."""
    graph, eucl, corner = [], [], []
    for group, label in zip(file_groups, labels):
        seed_obs = [json.load(open(p)) for p in ([group] if isinstance(group, str) else group)]
        for prefix, cast, bucket in [("Cxy_graph_r", int, graph),
                                     ("Cxy_euclidean_r", float, eucl),
                                     ("Cxy_corner_d", int, corner)]:
            got = _avg_over_seeds(seed_obs, prefix, cast)
            if got is not None:
                bucket.append((*got, label))
    return graph, eucl, corner


def _inset(ax, shape, g=2, n=3):
    ins = ax.inset_axes([0.55, 0.55, 0.42, 0.42])
    ins.set(xticks=[], yticks=[], aspect="equal")
    for s in ins.spines.values():
        s.set_visible(False)
    if shape == "triangular":
        adj, _ = triangular_gasket(g, n)
        x, y, _, _ = honeycomb_gasket_coordinates(g, n)
    else:
        adj, _ = honeycomb_gasket(g, n)
        xa, ya, xb, yb = honeycomb_gasket_coordinates(g, n)
        x, y = np.hstack((xa, xb)), np.hstack((ya, yb))
    nx.draw(nx.from_numpy_array(adj.toarray()), pos=dict(enumerate(zip(x, y))),
            ax=ins, node_size=0, edge_color="black", width=0.6)


def _pow10(v, _):
    if v <= 0 or not np.isclose(v, 10.0 ** round(np.log10(v))):
        return ""
    return {0: "$1$", 1: "$10$"}.get(round(np.log10(v)), rf"$10^{{{round(np.log10(v))}}}$")


_FRACS = {0.25: r"$\frac{1}{4}$", 0.5: r"$\frac{1}{2}$", 0.75: r"$\frac{3}{4}$", 1.5: r"$\frac{3}{2}$"}


def _yfmt(v, _):
    return next((s for k, s in _FRACS.items() if np.isclose(v, k)), rf"${v:g}$")


def _scatter_fit(ax, d, c, err, color, marker, filled, fit_lw, fit_ls="--"):
    ax.errorbar(d, c, yerr=err, fmt=marker, color=color, mfc=color if filled else "none",
                ms=4.5, lw=0.8, mew=0.85, capsize=2, capthick=1, ls="none", zorder=2)
    eta, A = _fit(d, c)
    if eta is not None:
        r = np.logspace(np.log10(d[c > 0].min()), np.log10(d[c > 0].max()), 200)
        pe = [patheffects.withStroke(linewidth=0.6, foreground="black")] if filled else None
        ax.plot(r, A * r ** -eta, color=color, lw=fit_lw, ls=fit_ls, alpha=0.85, zorder=3, path_effects=pe)
    return eta


def _corr_panel(ax, series, clean, xlabel, tag="", legend=False, xlim=None, ylim=None, yticks=None):
    handles = []
    for i in reversed(range(len(series))):  # smaller G on top
        d, c, err, label = series[i]
        _scatter_fit(ax, d, c, err, COLORS[i % len(COLORS)], MARKERS[i % len(MARKERS)], True, 1.2)
        handles.append(Line2D([0], [0], color=COLORS[i % len(COLORS)],
                              marker=MARKERS[i % len(MARKERS)], ls="none", ms=4, label=label))
    handles.reverse()
    for k, (d, c, err, label) in enumerate(clean):
        eta = _scatter_fit(ax, d, c, err, GREY[k % 2], GREY_MARK[k % 2], False, 1.5, fit_ls="-")
        handles.append(Line2D([0], [0], color=GREY[k % 2], marker=GREY_MARK[k % 2], mfc="none",
                              ms=4, label=label))

    ax.set(xscale="log", yscale="log")
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(r"$C_{xy}$", fontsize=16)
    ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.5)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if yticks:
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_yfmt))
        ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_pow10))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=14)

    if legend and handles:
        ax.legend(handles=handles, fontsize=11, loc="lower left", ncol=2, edgecolor="0.5",
                  columnspacing=0.4, labelspacing=0.15, handlelength=1.3)
    if tag:
        _tag(ax, tag)


def fig4(tri, hc, tri_clean=(), hc_clean=(), labels=None, clean_labels=("pristine lattice",),
         out="fig4", wspace=0.12):
    """C_xy vs graph distance d (top) and Euclidean distance r (bottom), with
    power-law fits and the clean lattice in grey."""
    labels = labels or [rf"$G={i + 1}$" for i in range(len(tri))]
    tg, te, tco = _load_correlators(tri, labels)
    hg, he, hco = _load_correlators(hc, labels)
    tcg, tce, tcc = _load_correlators(tri_clean, clean_labels)
    hcg, hce, hcc = _load_correlators(hc_clean, clean_labels)

    for shape, gd, ed, cd, prefix in [
        ("triangular", tg, te, tco, ""),
        ("honeycomb", hg, he, hco, ""),
        ("triangular", tcg, tce, tcc, "pristine "),
        ("honeycomb", hcg, hce, hcc, "pristine "),
    ]:
        for kind, sets in [
            ("network", gd),
            ("euclidean", ed),
            ("boundary", cd),
        ]:
            for d, c, _, lab in sets:
                eta, A = _fit(d, c)
                name = f"{prefix}{lab}"
                if eta is not None:
                    print(
                        f"{shape:10s} {name:18s} C_xy({kind}): "
                        f"eta = {eta:.4f}, A = {A:.4f}"
                    )
                else:
                    print(
                        f"{shape:10s} {name:18s} "
                        f"C_xy({kind}): no fit"
                    )

    fig, ax = plt.subplots(2, 2, figsize=(PAGE, 4/5*PAGE), sharex="col")
    tri_kw = dict(xlim=(0.6, 20), ylim=(0.4, 1.5), yticks=(0.5, 1.0))
    hc_kw = dict(xlim=(0.35, 45), ylim=(0.3, 1.6), yticks=(0.5, 1.0))
    _corr_panel(ax[0, 0], tg, tcg, r"$d$", "(a)", legend=True, **tri_kw)
    _corr_panel(ax[0, 1], hg, hcg, r"$d$", "(b)", **hc_kw)
    _corr_panel(ax[1, 0], te, tce, r"$r$", "(c)", **tri_kw)
    _corr_panel(ax[1, 1], he, hce, r"$r$", "(d)", **hc_kw)
    _inset(ax[0, 0], "triangular")
    _inset(ax[0, 1], "honeycomb")
    ax[0, 1].set_ylabel("")
    ax[1, 1].set_ylabel("")

    fig.tight_layout(pad=0.15)
    fig.subplots_adjust(wspace=wspace)
    _save(fig, out)



def _bench_series(rows, shape, lattice):
    """[(N, E/N, err, E_ed/N, rel_err)] averaged over seeds."""
    by_n = {}
    for r in rows:
        if r["shape"] == shape and r["lattice"] == lattice:
            by_n.setdefault(r["N"], []).append(r)
    out = []
    for N, items in sorted(by_n.items()):
        e = np.array([it["E_vmc_per_site"] for it in items])
        ed = [it["E_ed_per_site"] for it in items if it.get("E_ed_per_site") is not None]
        rel = [it["rel_error"] for it in items if it.get("rel_error") is not None]
        err = e.std() if len(e) > 1 else np.mean([it["E_vmc_err"] for it in items]) / N
        out.append((N, e.mean(), err, np.mean(ed) if ed else None, np.mean(rel) if rel else None))
    return out


def fig5(files, out="fig5"):
    """RBM on gaskets vs clean lattices: E/N vs 1/N (ED as x) and the relative
    error vs N. `files` are *_benchmark.json from run_rbm_bench.py."""
    rows = [json.load(open(f)) for f in files]
    if not rows:
        print("fig5: no benchmark files; skipping")
        return

    fig, ax = plt.subplots(1, 3, figsize=(PAGE, PAGE * 0.34))
    styles = [("clean", COLORS[0], "o", None), ("gasket", COLORS[3], "s", "none")]

    for a, shape, title in zip(ax[:2], ["triangular", "honeycomb"], ["Triangular", "Honeycomb"]):
        for lattice, color, marker, fill in styles:
            s = _bench_series(rows, shape, lattice)
            if not s:
                continue
            N, e, err, ed, _ = zip(*s)
            a.errorbar([1 / n for n in N], e, yerr=err, marker=marker, mfc=fill, color=color,
                       ms=4, lw=0.9, capsize=2, label=lattice)
            pts = [(1 / n, v) for n, v in zip(N, ed) if v is not None]
            if pts:
                a.plot(*zip(*pts), "x", color=color, ms=5, mew=1.1, ls="none", zorder=5)
        a.set_title(title, fontsize=12)
        a.set_xlabel(r"$1/N$", fontsize=12)
        a.grid(alpha=0.3, ls="--", lw=0.4)
        a.tick_params(direction="in", top=True, right=True, labelsize=12)
    ax[0].set_ylabel(r"$E/N$", fontsize=12)
    ax[0].legend(fontsize=11, loc="best", edgecolor="0.5")

    for shape, marker in [("triangular", "o"), ("honeycomb", "^")]:
        for lattice, color in [("clean", COLORS[0]), ("gasket", COLORS[3])]:
            pts = [(N, rel) for N, *_, rel in _bench_series(rows, shape, lattice) if rel is not None]
            if pts:
                ax[2].plot(*zip(*pts), marker=marker, color=color, ms=4, lw=0.9,
                           label=f"{shape[:3]}. {lattice}")
    ax[2].set_yscale("log")
    ax[2].set_xlabel(r"$N$", fontsize=12)
    ax[2].set_ylabel(r"$|E_\mathrm{VMC} - E_\mathrm{ED}| / |E_\mathrm{ED}|$", fontsize=12)
    ax[2].grid(alpha=0.3, ls="--", lw=0.4, which="both")
    ax[2].tick_params(direction="in", top=True, right=True, labelsize=12)
    if ax[2].get_legend_handles_labels()[0]:
        ax[2].legend(fontsize=11, loc="best", edgecolor="0.5")

    for a, tag in zip(ax, ["(a)", "(b)", "(c)"]):
        _tag(a, tag)
    fig.tight_layout(pad=0.3)
    _save(fig, out)



def _fidelity_matrix(path):
    """Seed x seed matrix of F (NaN diagonal) from a fidelity.py JSON."""
    if path is None:
        return None, None
    try:
        d = json.load(open(path))
    except FileNotFoundError:
        return None, None
    seeds = d["seeds"]
    idx = {s: i for i, s in enumerate(seeds)}
    F = np.full((len(seeds), len(seeds)), np.nan)
    for key, v in d["fidelities"].items():
        a, b = (int(p.replace("seed", "")) for p in key.split("_"))
        F[idx[a], idx[b]] = F[idx[b], idx[a]] = v
    return F, seeds


def fig6(tri_files, hc_files, gens, out="fig6", vmin=1e-5, cmap="viridis"):
    """1 - F between seeds, one heatmap per (shape, G). F < 0 means the MC
    estimate failed; those entries are hatched."""
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle

    norm = LogNorm(vmin=vmin, vmax=1)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("white")
    ncol = len(gens)
    width = PAGE if ncol > 2 else PAGE / 2
    fig, ax = plt.subplots(2, ncol, figsize=(width, 3.0 if ncol <= 2 else PAGE * 0.47), squeeze=False)

    for r, (files, shape) in enumerate([(tri_files, "Triangular"), (hc_files, "Honeycomb")]):
        for c, (g, path) in enumerate(zip(gens, files)):
            a = ax[r, c]
            F, seeds = _fidelity_matrix(path)
            if F is None:
                a.axis("off")
                continue
            n = len(seeds)
            bad = F < 0
            a.imshow(np.where(bad, np.nan, np.clip(1 - F, vmin, None)), cmap=cm, norm=norm,
                     interpolation="nearest", origin="upper")
            for i, j in zip(*np.nonzero(bad)):
                a.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc="0.85", ec="0.4",
                                      hatch="////", lw=0))

            a.set_xticks(range(n))
            a.set_yticks(range(n))
            a.set_xticklabels(seeds if r == 1 else [], fontsize=8)
            a.set_yticklabels(seeds if c == 0 else [], fontsize=8)
            a.tick_params(which="both", length=0, top=False, right=False)
            a.minorticks_off()
            if r == 0:
                a.set_title(rf"$G={g}$", fontsize=11, pad=4)
        ax[r, 0].set_ylabel("seed", fontsize=10)
    for a in ax[1]:
        a.set_xlabel("seed", fontsize=10)

    fig.tight_layout(pad=0.2, w_pad=0.3, h_pad=0.4)
    right = 0.88 if ncol > 2 else 0.78
    fig.subplots_adjust(right=right, wspace=0.12)
    cax = fig.add_axes([right + 0.025, ax[1, -1].get_position().y0, 0.018,
                        ax[0, -1].get_position().y1 - ax[1, -1].get_position().y0])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), cax=cax, extend="min")
    cb.set_label(r"$1-F$", fontsize=11)
    cb.ax.tick_params(labelsize=9, direction="in")
    _save(fig, out)


if __name__ == "__main__":
    gens = [1, 2, 3, 4, 5]

    fig1()
    fig2("img/vit_fractal.pdf")

    tri_history = [
        "data/ViT/triangular/vit_triangular_G=1_seed=1_energy_history.json",
        "data/ViT/triangular/vit_triangular_G=2_seed=1_energy_history.json",
        "data/ViT/triangular/vit_triangular_G=3_seed=1_energy_history.json",
        "data/ViT/triangular/vit_triangular_G=4_seed=1_energy_history.json",
        "data/ViT/triangular/vit_triangular_G=5_seed=1_energy_history.json",
    ]
    hc_history = [
        "data/ViT/honeycomb/vit_honeycomb_G=1_seed=1_energy_history.json",
        "data/ViT/honeycomb/vit_honeycomb_G=2_seed=1_energy_history.json",
        "data/ViT/honeycomb/vit_honeycomb_G=3_seed=1_energy_history.json",
        "data/ViT/honeycomb/vit_honeycomb_G=4_seed=1_energy_history.json",
        "data/ViT/honeycomb/vit_honeycomb_G=5_seed=1_energy_history.json"
    ]
    fig3(tri_history, hc_history, gens)

    def seed_obs(shape, g):
        return sorted(glob(f"data/ViT/{shape}/vit_{shape}_G={g}_seed=*_symm_observables.json"))

    def clean_ref(shape):
        for L in (15, 13, 9):  # largest clean size available
            g = sorted(glob(f"data/RBM_bench/rbm_{shape}_clean_L={L}_seed=*_observables.json"))
            if g:
                return [g]
        return []

    fig4([seed_obs("triangular", g) for g in gens],
         [seed_obs("honeycomb", g) for g in gens],
         tri_clean=clean_ref("triangular"),
         hc_clean=clean_ref("honeycomb"),
         labels=[rf"$G={g}$" for g in gens])

    fig5(sorted(glob("data/RBM_bench/*_benchmark.json")))

    def fid_file(shape, g, symm):
        return f"data/ViT/{shape}/fidelity_{shape}_gasket_G={g}{'_symm' if symm else ''}.json"

    fid_gens = [4, 5]
    fid_symm = {4: True, 5: False}  # no projected states at G=5
    fig6([fid_file("triangular", g, fid_symm[g]) for g in fid_gens],
         [fid_file("honeycomb", g, fid_symm[g]) for g in fid_gens], fid_gens)
