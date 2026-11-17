"""Figures of pp.py rebuilt from the S^z-sector states of trainViT_sz.py (data/ViT_sz).
`python pp_sz.py` writes the paper figures img/fig1.pdf, img/fig2.pdf, img/fig3_sz_vscore.pdf,
img/fig4_sz.pdf and img/fig5_sz.pdf. fig2 is built from img/vit_fractal.pdf.
"""

import json
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from pp import COLORS, PAGE, _load_history, _n_sites, _save, _tag, fig1, fig2, fig4

ROOT = "data/ViT_sz"

def plot_seed(shape, g, among=None):
    """Seed whose training history fig3_vscore draws: the lowest final energy of the retrained states
    (the best variational state), among the seeds `among` if given. 1 if none has metadata."""
    energies = {}
    for s in among or SEEDS:
        path = f"{ROOT}/{shape}/vit_{shape}_G={g}_seed={s}_symm_metadata.json"
        if os.path.exists(path):
            energies[s] = json.load(open(path))["final_results"]["final_energy"]
    return min(energies, key=energies.get) if energies else 1


_load_run_history = _load_history


def _load_history(path):
    """pp._load_history, with the runs that trainViT_sz.py --continue renamed to *_prev in front:
    their main and C3v steps both go into the main stage, since training went on after them."""
    E, V, start, sE, sV = _load_run_history(path)
    prev = path.replace("_energy_history.json", "_prev_energy_history.json")
    if prev != path and os.path.exists(prev):
        pE, pV, _, psE, psV = _load_history(prev)
        E, V = np.concatenate([pE, psE, E]), np.concatenate([pV, psV, V])
        start = len(E)
    return E, V, start, sE, sV
LAST_ITERS = 200
SEEDS = [1, 2, 3, 4, 5]
FAILED_F = 1.01       # a fidelity estimate above this, or below 0, failed: sampling noise stays far below
RESOLVED_SIGMA = 2    # a Monte Carlo energy difference above this many sigma gets a dot


def fig3_vscore(tri_files, hc_files, gens, labels=None, out="fig3_sz_vscore", last=LAST_ITERS, skip=1):
    """V-score N Var(E) / E^2 (Wu et al., Science 2024) over the last `last` iterations,
    right-aligned as in fig3: one panel per shape, dashed after the C3v projection. It is
    zero only for an eigenstate and does not grow with N, so every G shares one axis. skip drops
    the smallest generations (G=1 is summed exactly and falls to about 1e-12)."""
    labels = labels or [rf"$G={g}$" for g in gens]
    fig, ax = plt.subplots(1, 2, figsize=(PAGE, PAGE * 0.36), sharex=True, sharey=True)
    for a, files, shape in ((ax[0], tri_files, "triangular"), (ax[1], hc_files, "honeycomb")):
        for i, (path, g, label) in enumerate(zip(files, gens, labels)):
            if i < skip or path is None:
                continue
            E, V, start, sE, sV = _load_history(path)
            N = _n_sites(shape, g)
            v = N * np.clip(V, 1e-14, None) / E**2
            sv = N * np.clip(sV, 1e-14, None) / sE**2 if len(sE) else sE
            offset = start + len(sv) - last
            color = COLORS[i % len(COLORS)]
            it = np.arange(len(v))
            keep = it >= offset
            a.plot(it[keep] - offset, v[keep], color=color, lw=2.2, label=label)
            if len(sv):
                a.plot(np.arange(start, start + len(sv)) - offset, sv, color=color, lw=2.2, ls="--")
                a.axvline(start - offset, color=color, lw=0.7, ls="--", alpha=0.35, zorder=0)
        a.set_yscale("log")
        a.grid(alpha=0.3, ls="--", lw=0.4)
        a.tick_params(direction="in", top=True, right=True, labelsize=12)
        a.set_xlabel(f"Iteration (last {last})", fontsize=12)
    ax[0].set_xlim(0, last)
    ax[0].set_ylabel("V-score", fontsize=12)   # N Var(E) / E^2: define it in the caption
    ax[0].legend(fontsize=10, ncol=2, loc="upper right", columnspacing=1.0, handlelength=1.5)
    ax[0].set_xticks([0, 50, 100, 150])
    for a, tag, name in zip(ax, ["(a)", "(b)"], ["Triangular", "Honeycomb"]):
        _tag(a, tag, x=0.09)
        a.set_title(name, fontsize=12)
    fig.tight_layout(pad=0.25)
    fig.subplots_adjust(wspace=0.06)
    _save(fig, out)


def _existing(path):
    return path if os.path.exists(path) else None


def _ramp(name):
    """Light-to-dark colormap without the near-white end the empty diagonal uses."""
    return ListedColormap(plt.get_cmap(name)(np.linspace(0.15, 1, 256)))


def _seed_energies(shape, g, fid, seeds):
    """{seed: (E, error)}: the exact sums of sector_fidelity.py where it has them, else the final
    energies of trainViT_sz.py. None if a state is missing or only a promoted checkpoint
    (promote_ckpt.py), whose stored energy is a training average and not an evaluation."""
    if fid is not None and fid.get("exact"):
        return {s: (fid["energies"][str(s)]["mean"], 0.0) for s in seeds}
    out = {}
    for s in seeds:
        path = f"{ROOT}/{shape}/vit_{shape}_G={g}_seed={s}_symm_metadata.json"
        if not os.path.exists(path):
            return None
        meta = json.load(open(path))
        if "promoted_from_checkpoint" in meta:
            return None
        r = meta["final_results"]
        out[s] = (r["final_energy"], r["final_energy_error"])
    return out


def _fidelity_cells(fid, seeds):
    """Lower triangle of 1 - F, at least its error since a smaller value cannot be resolved, and
    the mask of failed estimates (NaN in the matrix)."""
    idx = {s: i for i, s in enumerate(seeds)}
    values = np.full((len(seeds), len(seeds)), np.nan)
    failed = np.zeros_like(values, dtype=bool)
    for key, F in fid["fidelities"].items():
        a, b = (idx[int(p.replace("seed", ""))] for p in key.split("_"))
        i, j = max(a, b), min(a, b)
        if F < 0 or F > FAILED_F:
            failed[i, j] = True
        else:
            values[i, j] = max(1 - F, fid["fidelity_errors"].get(key, 0.0))
    return values, failed


def _energy_cells(energies, seeds, n_sites):
    """Upper triangle of |E_i - E_j| / N and the mask of Monte Carlo pairs that differ by more than
    RESOLVED_SIGMA. Exact energies have no error, so they are not marked."""
    values = np.full((len(seeds), len(seeds)), np.nan)
    resolved = np.zeros_like(values, dtype=bool)
    for i, a in enumerate(seeds):
        for j, b in enumerate(seeds):
            if j > i:
                diff = abs(energies[a][0] - energies[b][0])
                noise = RESOLVED_SIGMA * np.hypot(energies[a][1], energies[b][1])
                values[i, j] = diff / n_sites
                resolved[i, j] = noise > 0 and diff >= noise
    return values, resolved


def fig5(gens, out="fig5_sz",f_range=(1e-5, 1), e_range=(1e-7, 1e-4)):
    """Do the seeds reach the same state, and the same energy? One panel per G, triangular on top,
    honeycomb below. Lower triangle: 1 - F. Upper triangle: |Delta E| / N, most of it sampling noise
    at G >= 2 unless dotted. Both are darker for a larger difference; values outside the range are
    drawn at its ends."""
    f_norm, f_cmap = LogNorm(*f_range), _ramp("YlGnBu")
    e_norm, e_cmap = LogNorm(*e_range), _ramp("OrRd")
    ncol = len(gens)
    fig, ax = plt.subplots(2, ncol, figsize=(PAGE, 3.62), squeeze=False)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.93, bottom=0.27, wspace=0.1, hspace=0.1)

    any_failed = False
    clipped = {"f": [False, False], "e": [False, False]}   # data below / above each colour range
    for r, shape in enumerate(["triangular", "honeycomb"]):
        for c, g in enumerate(gens):
            a = ax[r, c]
            path = f"{ROOT}/{shape}/fidelity_{shape}_gasket_G={g}_sector_symm.json"
            fid = json.load(open(path)) if os.path.exists(path) else None
            seeds = fid["seeds"] if fid else SEEDS
            n = len(seeds)
            energies = _seed_energies(shape, g, fid, seeds)

            a.imshow(np.full((n, n), np.nan), vmin=0, vmax=1)   # sets the extent when a panel is empty
            if fid is not None:
                values, failed = _fidelity_cells(fid, seeds)
                any_failed |= bool(failed.any())
                clipped["f"][0] |= bool(np.nanmin(values, initial=np.inf) < f_range[0])
                clipped["f"][1] |= bool(np.nanmax(values, initial=-np.inf) > f_range[1])
                a.imshow(np.clip(values, f_range[0], None), cmap=f_cmap, norm=f_norm, interpolation="nearest")
                for i, j in zip(*np.nonzero(failed)):
                    a.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc="0.85", ec="0.4", hatch="////", lw=0))
            if energies is not None:
                values, resolved = _energy_cells(energies, seeds, _n_sites(shape, g))
                clipped["e"][0] |= bool(np.nanmin(values, initial=np.inf) < e_range[0])
                clipped["e"][1] |= bool(np.nanmax(values, initial=-np.inf) > e_range[1])
                a.imshow(np.clip(values, *e_range), cmap=e_cmap, norm=e_norm, interpolation="nearest")
                rows, cols = np.nonzero(resolved)
                a.plot(cols, rows, "o", ms=2.5, color="k", ls="none")
            if fid is None and energies is None:
                a.text(0.5, 0.5, "pending", transform=a.transAxes, ha="center", va="center",
                       fontsize=9, color="0.4")

            a.set_xlim(-0.5, n - 0.5)
            a.set_ylim(n - 0.5, -0.5)
            a.set_xticks(range(n))
            a.set_yticks(range(n))
            a.set_xticklabels(seeds if r == 1 else [], fontsize=8)
            a.set_yticklabels(seeds if c == 0 else [], fontsize=8)
            # white gaps between the cells
            a.set_xticks(np.arange(-0.5, n), minor=True)
            a.set_yticks(np.arange(-0.5, n), minor=True)
            a.grid(which="minor", color="white", lw=1.2)
            a.grid(which="major", visible=False)
            a.set_axisbelow(False)
            a.tick_params(which="both", length=0, top=False, right=False)
            a.spines[:].set_visible(False)
            if r == 0:
                a.set_title(rf"$G={g}$", fontsize=11, pad=4)
            if r == 1:
                a.set_xlabel("seed", fontsize=10)
        ax[r, 0].set_ylabel("seed", fontsize=10)
        ax[r, 0].annotate(shape.capitalize(), xy=(0, 0.5), xycoords="axes fraction", xytext=(-32, 0),
                          textcoords="offset points", rotation=90, ha="right", va="center", fontsize=11)

    # Colour scales and key below all panels, so that neither scale reads as belonging to one row,
    # centred as a group under the panels. An arrow marks an end where data are drawn clipped.
    ext = lambda lo, hi: {(False, False): "neither", (True, False): "min",
                          (False, True): "max", (True, True): "both"}[(lo, hi)]
    bar_w, gap, key_w = 0.25, 0.06, 0.15
    x0 = (0.085 + 0.995) / 2 - (2 * bar_w + 2 * gap + key_w) / 2
    for x, norm, cmap, label, extend in [
            (x0, f_norm, f_cmap, r"$1-F$ (lower triangle)", ext(*clipped["f"])),
            (x0 + bar_w + gap, e_norm, e_cmap, r"$|\Delta E|/N$ (upper triangle)", ext(*clipped["e"]))]:
        cax = fig.add_axes([x, 0.075, bar_w, 0.03])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                          extend=extend)
        cb.set_label(label, fontsize=10, labelpad=3)
        cb.ax.xaxis.set_label_position("top")
        cb.ax.tick_params(labelsize=8, direction="in")
    # the "failed estimate" key only when a panel has one
    handles = [Patch(fc="0.85", ec="0.4", hatch="////", lw=0, label="failed estimate")] if any_failed else []
    handles.append(Line2D([], [], marker="o", ms=2.5, color="k", ls="none",
                          label=rf"$\Delta E$ above ${RESOLVED_SIGMA}\sigma$"))
    fig.legend(handles=handles,
               loc="center left", bbox_to_anchor=(x0 + 2 * (bar_w + gap) - 0.01, 0.09), frameon=False,
               fontsize=9, handletextpad=0.3, borderaxespad=0)
    _save(fig, out)


if __name__ == "__main__":
    gens = [1, 2, 3, 4, 5]

    fig1()
    fig2("img/vit_fractal.pdf")

    def history(shape, g):
        seed = plot_seed(shape, g)
        return _existing(f"{ROOT}/{shape}/vit_{shape}_G={g}_seed={seed}_symm_energy_history.json")

    fig3_vscore([history("triangular", g) for g in gens], [history("honeycomb", g) for g in gens], gens)

    def seed_obs(shape, g):
        return sorted(glob(f"{ROOT}/{shape}/vit_{shape}_G={g}_seed=*_symm_observables.json"))

    def clean_ref(shape):
        for L in (15, 13, 9):  # largest clean size available
            g = sorted(glob(f"data/RBM_bench/rbm_{shape}_clean_L={L}_seed=*_observables.json"))
            if g:
                return [g]
        return []

    # only generations with observables, so colours and labels stay matched
    tri_gens = [g for g in gens if seed_obs("triangular", g)]
    hc_gens = [g for g in gens if seed_obs("honeycomb", g)]
    obs_gens = [g for g in gens if g in tri_gens or g in hc_gens]
    fig4([seed_obs("triangular", g) for g in obs_gens],
         [seed_obs("honeycomb", g) for g in obs_gens],
         tri_clean=clean_ref("triangular"),
         hc_clean=clean_ref("honeycomb"),
         labels=[rf"$G={g}$" for g in obs_gens], out="fig4_sz")

    fig5(gens)
