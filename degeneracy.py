import argparse
import json
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

from fractals import honeycomb_gasket, triangular_gasket

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except ImportError:
    pass
plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif"})

PAGE = 6.75
SHAPES = ["triangular", "honeycomb"]
GENS = [1, 2, 3, 4, 5]
N_PATCH = 2
F_TOL = 1e-3     # |F| excursions beyond [0, 1] smaller than this count as noise
VMIN = 1e-5


def _base(shape, G, seed):
    return f"data/ViT/{shape}/vit_{shape}_G={G}_seed={seed}"


def _n_sites(shape, G):
    build = triangular_gasket if shape == "triangular" else honeycomb_gasket
    return build(G, N_PATCH)[0].shape[0]


def _seeds(shape, G):
    files = glob(f"{_base(shape, G, '*')}_metadata.json")
    return sorted(int(f.split("seed=")[1].split("_")[0]) for f in files)


def _obs_file(shape, G, seed, symm):
    """Observables JSON for this seed: expect.py output, else the --compute cache."""
    tag = "_symm" if symm else ""
    for f in (f"{_base(shape, G, seed)}{tag}_observables.json", f"{_base(shape, G, seed)}{tag}_sz.json"):
        if os.path.exists(f):
            return f
    return None


def load_seed(shape, G, seed, symm=False):
    """(E, dE, Sz, dSz, E_from_obs) for one seed; Sz is None when not measured."""
    N = _n_sites(shape, G)
    meta = json.load(open(f"{_base(shape, G, seed)}_metadata.json"))
    path = _obs_file(shape, G, seed, symm)
    if path is not None:
        o = json.load(open(path))
        E, dE = o["E"]["mean"], o["E"]["error"]
        Sz, dSz = N * o["Mz"]["mean"] / 2, N * o["Mz"]["error"] / 2
        return E, dE, Sz, dSz, True
    # Fall back to the last training energy; the error ignores autocorrelation.
    res = meta["final_results"]
    key = "symm_final_" if symm else "final_"
    E = res[f"{key}energy"]
    dE = np.sqrt(res[f"{key}variance"] / meta["sampler"]["n_samples"])
    return E, dE, None, None, False


def load_fidelity(shape, G, seeds, symm=False):
    """Seed x seed matrix of measured F (NaN where missing), or None."""
    path = f"data/ViT/{shape}/fidelity_{shape}_gasket_G={G}{'_symm' if symm else ''}.json"
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    idx = {s: i for i, s in enumerate(seeds)}
    F = np.full((len(seeds), len(seeds)), np.nan)
    for key, v in d["fidelities"].items():
        a, b = (int(p.replace("seed", "")) for p in key.split("_"))
        if a in idx and b in idx:
            F[idx[a], idx[b]] = F[idx[b], idx[a]] = v
    return F


def doublet_bound(Sz):
    """Largest F between two states of the S^z = +-1/2 doublet with these <S^z>."""
    p = np.clip(0.5 + np.asarray(Sz, dtype=float), 0, 1)
    return (np.sqrt(np.outer(p, p)) + np.sqrt(np.outer(1 - p, 1 - p))) ** 2


def _energy_panel(ax, seeds, E, dE, from_obs):
    dE_best = E - np.min(E)
    for x, y, err, filled in zip(seeds, dE_best, dE, from_obs):
        ax.errorbar(x, y, yerr=err, fmt="o", ms=4, color="#0072B2", capsize=2,
                    mfc="#0072B2" if filled else "white")
    ax.axhline(0, color="0.6", lw=0.6, zorder=0)
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.get_offset_text().set_fontsize(7)


def _sz_panel(ax, seeds, Sz, dSz, odd):
    for v in ([0.5, -0.5] if odd else [0.0]):
        ax.axhline(v, color="0.6", ls="--", lw=0.8, zorder=0)
    missing = []
    for x, y, err in zip(seeds, Sz, dSz):
        if y is None:
            missing.append(str(x))
        else:
            ax.errorbar(x, y, yerr=err, fmt="s", ms=4, color="#D55E00", capsize=2)
    if missing:
        ax.text(0.5, 0.03, "no $M_z$: " + ",".join(missing), transform=ax.transAxes,
                ha="center", va="bottom", fontsize=6, color="0.4")
    ax.set_ylim(-0.8, 0.8)


def _fidelity_panel(ax, F, Sz, odd, cmap):
    n = F.shape[0]
    lower = np.tril(np.ones((n, n), bool), -1)
    upper = lower.T
    bad = lower & ((F < -F_TOL) | (F > 1 + F_TOL))

    M = np.full((n, n), np.nan)
    M[lower & ~bad] = 1 - np.clip(F[lower & ~bad], 0, 1)
    if odd and all(s is not None for s in Sz):
        M[upper] = 1 - doublet_bound(Sz)[upper]
    ax.imshow(np.clip(M, VMIN, None), cmap=cmap, norm=LogNorm(VMIN, 1),
              interpolation="nearest", origin="upper")
    for i, j in zip(*np.nonzero(bad)):
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc="0.85", ec="0.4", hatch="////", lw=0))
    ax.plot([-0.5, n - 0.5], [-0.5, n - 0.5], color="k", lw=0.5)


def figure(shape, gens=GENS, symm=False, out=None, cmap="viridis"):
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("white")
    fig, ax = plt.subplots(3, len(gens), figsize=(PAGE, 4.6), squeeze=False,
                           gridspec_kw={"height_ratios": [1, 1, 1.25]})

    for c, G in enumerate(gens):
        seeds = _seeds(shape, G)
        N = _n_sites(shape, G)
        odd = N % 2 == 1
        rows = [load_seed(shape, G, s, symm) for s in seeds]
        E, dE, Sz, dSz, from_obs = map(list, zip(*rows))

        _energy_panel(ax[0, c], seeds, np.array(E), np.array(dE), from_obs)
        _sz_panel(ax[1, c], seeds, Sz, dSz, odd)
        ax[0, c].set_title(rf"$G={G}$" + "\n" + rf"$N={N}$" + (" (odd)" if odd else ""), fontsize=9)

        F = load_fidelity(shape, G, seeds, symm)
        if F is None:
            ax[2, c].axis("off")
        else:
            _fidelity_panel(ax[2, c], F, Sz, odd, cm)
            ax[2, c].set_xticks(range(len(seeds)), seeds, fontsize=7)
            ax[2, c].set_yticks(range(len(seeds)), seeds, fontsize=7)
            ax[2, c].tick_params(length=0)
            ax[2, c].minorticks_off()
            ax[2, c].set_xlabel("seed", fontsize=8)

        for a in ax[:2, c]:
            a.set_xticks(seeds)
            a.set_xlim(min(seeds) - 0.6, max(seeds) + 0.6)
            a.tick_params(labelsize=7)
            a.minorticks_off()
        ax[1, c].set_xlabel("seed", fontsize=8)

    ax[0, 0].set_ylabel(r"$E - E_{\rm best}$", fontsize=9)
    ax[1, 0].set_ylabel(r"$S^z = N M_z / 2$", fontsize=9)
    ax[2, 0].set_ylabel("seed", fontsize=8)
    fig.suptitle(f"{shape.capitalize()} gasket{' (symmetry projected)' if symm else ''}", fontsize=10)

    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.4)
    fig.subplots_adjust(right=0.89)
    pos = ax[2, -1].get_position()
    cax = fig.add_axes([0.91, pos.y0, 0.015, pos.height])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=LogNorm(VMIN, 1), cmap=cm), cax=cax, extend="min")
    cb.set_label(r"$1-F$", fontsize=9)
    cb.ax.tick_params(labelsize=7)
    fig.text(0.5, -0.01, r"$1-F$: lower triangle measured, upper triangle smallest value allowed by "
             r"$S^z$ within the $\pm\frac{1}{2}$ doublet (odd $N$).  Open markers: energy from training, "
             r"no observables.  Hatched: failed MC estimate.", ha="center", va="top", fontsize=6.5, wrap=True)

    out = out or f"img/degeneracy_{shape}{'_symm' if symm else ''}.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def _sector_json(root, shape, G):
    """sector_fidelity.py output for (shape, G) under root, C3v version preferred."""
    for tag in ("_symm", ""):
        path = f"{root}/{shape}/fidelity_{shape}_gasket_G={G}_sector{tag}.json"
        if os.path.exists(path):
            return json.load(open(path))
    return None


def _trained_energies(root, shape, G, seeds):
    """Final energies that trainViT_sz.py stored in the metadata of each retrained state,
    evaluated on its equilibrated training chains."""
    out = {}
    for s in seeds:
        base = f"{root}/{shape}/vit_{shape}_G={G}_seed={s}"
        for meta in (f"{base}_symm_metadata.json", f"{base}_metadata.json"):
            if os.path.exists(meta):
                r = json.load(open(meta))["final_results"]
                out[str(s)] = {"mean": r["final_energy"], "error": r["final_energy_error"],
                               "variance": r["final_variance"]}
                break
    return out


def _sector_matrix(d, seeds):
    """F between seeds; for Monte Carlo results F is capped at 1 - (its error), since a
    smaller 1 - F cannot be resolved"""
    idx = {s: i for i, s in enumerate(seeds)}
    F = np.full((len(seeds), len(seeds)), np.nan)
    errors = d.get("fidelity_errors", {}) if not d.get("exact") else {}
    for key, v in d["fidelities"].items():
        a, b = (int(p.replace("seed", "")) for p in key.split("_"))
        if a in idx and b in idx:
            F[idx[a], idx[b]] = F[idx[b], idx[a]] = min(v, 1 - errors.get(key, 0.0))
    return F


def figure_retrained(shape, old_root="data/ViT", new_root="data/ViT_sz", out=None, cmap="viridis",
                     floor=1e-12, gens=GENS):
    """Before/after the fixed-S^z retraining, for every G that has sector results in old_root or
    new_root (a G with only one of them shows only that one).

    top:    E - E_ref per seed; E_ref is the exact sector energy when known,
            else the lowest energy found. Grey: original states projected onto the
            sector (sector_fidelity.py), blue: retrained (sector_fidelity.py where it sums
            exactly, else the final energies of the training metadata).
    bottom: 1 - F between seeds; lower triangle original, upper triangle retrained.
    """
    gens = [G for G in gens if _sector_json(new_root, shape, G) or _sector_json(old_root, shape, G)]
    if not gens:
        print(f"no sector results for {shape}; run sector_fidelity.py first")
        return
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("white")
    norm = LogNorm(floor, 1)
    fig, ax = plt.subplots(2, len(gens), figsize=(max(2.2 * len(gens), 3.4), 4.0), squeeze=False,
                           gridspec_kw={"height_ratios": [1, 1.1]})

    for c, G in enumerate(gens):
        new = _sector_json(new_root, shape, G)
        old = _sector_json(old_root, shape, G)
        ref = new or old
        seeds = sorted({s for d in (old, new) if d for s in d["seeds"]})
        if new is not None and not new.get("exact"):   # exact sums beat the training's MC estimate
            new = {**new, "energies": {**new["energies"], **_trained_energies(new_root, shape, G, seeds)}}
        ed = ref.get("ed_sector_energies")
        ed_file = f"data/ED/{shape}_G={G}_sz={ref['sector_sz']:+g}.json"   # ed_sector.py
        if not ed and os.path.exists(ed_file):
            ed = json.load(open(ed_file))["energies"]
        e_ref = ed[0] if ed else min(e["mean"] for d in (old, new) if d for e in d["energies"].values())

        a = ax[0, c]
        for d, dx, color, marker, label in [(old, -0.12, "0.55", "s", "original, in sector"),
                                            (new, +0.12, "#0072B2", "o", "retrained")]:
            if d is None:
                continue
            have = [s for s in seeds if str(s) in d["energies"]]
            e = [d["energies"][str(s)] for s in have]
            y = np.array([x["mean"] - e_ref for x in e])
            if d.get("exact"):
                y = np.maximum(y, floor)
            a.errorbar(np.array(have) + dx, y, yerr=[x["error"] for x in e], fmt=marker, ms=4,
                       color=color, capsize=2, label=label)
        if all(d.get("exact") for d in (old, new) if d):
            a.set_yscale("log")
            a.set_ylim(floor / 3, 3)
        else:
            # Monte Carlo: E - E_ref is noise around zero and can be negative
            a.set_yscale("symlog", linthresh=1e-5)
            a.set_ylim(-1e-3, 3)
            a.axhline(0, color="k", lw=0.5)
        a.set_xticks(seeds)
        a.tick_params(labelsize=7)
        a.set_title(rf"$G={G}$, $N={ref['n_sites']}$" + "\n" + rf"$S^z={ref['sector_sz']:+g}$".replace("+0$", "0$"), fontsize=9)
        a.set_xlabel("seed", fontsize=8)

        b = ax[1, c]
        n = len(seeds)
        lower = np.tril(np.ones((n, n), bool), -1)
        M = np.full((n, n), np.nan)
        if old is not None:
            Fo = _sector_matrix(old, seeds)
            M[lower] = 1 - np.clip(Fo[lower], 0, 1)
        if new is not None:
            Fn = _sector_matrix(new, seeds)
            M[lower.T] = 1 - np.clip(Fn[lower.T], 0, 1)
        b.imshow(np.clip(M, floor, None), cmap=cm, norm=norm, interpolation="nearest", origin="upper")
        b.plot([-0.5, n - 0.5], [-0.5, n - 0.5], color="k", lw=0.5)
        b.set_xticks(range(n), seeds, fontsize=7)
        b.set_yticks(range(n), seeds, fontsize=7)
        b.tick_params(length=0)
        b.minorticks_off()
        b.set_xlabel("seed", fontsize=8)

    ax[0, 0].set_ylabel(r"$E - E_{\rm ED}$", fontsize=9)
    ax[1, 0].set_ylabel("seed", fontsize=8)
    ax[0, 0].legend(fontsize=6, loc="center", frameon=False)
    fig.suptitle(f"{shape.capitalize()} gasket: fixed-$S^z$ retraining", fontsize=10)

    fig.tight_layout(pad=0.3, h_pad=0.8, w_pad=0.5)
    fig.subplots_adjust(right=0.84)
    pos = ax[1, -1].get_position()
    cax = fig.add_axes([0.87, pos.y0, 0.025, pos.height])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), cax=cax, extend="min")
    cb.set_label(r"$1-F$", fontsize=9)
    cb.ax.tick_params(labelsize=7)
    fig.text(0.5, -0.01, r"$1-F$: lower triangle original states projected onto the sector, "
             "upper triangle retrained. Values below the floor are drawn at the floor; "
             "Monte Carlo values (G without exact sums) at no less than their error.",
             ha="center", va="top", fontsize=6.5, wrap=True)

    out = out or f"img/retrained_{shape}.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def compute_missing(gens=GENS, symm=False, n_samples=2**13):
    """Measure E and M_z for seeds that have no observables file yet."""
    import netket as nk
    from netket.operator.spin import sigmaz

    from expect import load_vstate_vit, load_vstate_vit_symm
    from trainViT import build_fractal, patch_distances
    from utils import build_hamiltonian

    for shape in SHAPES:
        for G in gens:
            missing = [s for s in _seeds(shape, G) if _obs_file(shape, G, s, symm) is None]
            if not missing:
                continue
            g, patches, x, y = build_fractal(shape, G, N_PATCH)
            distances = patch_distances(patches, x, y)
            hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
            ops = {"E": build_hamiltonian(hi, g),
                   "Mz": sum(sigmaz(hi, i) for i in range(g.n_nodes)) / g.n_nodes}
            for seed in missing:
                base = _base(shape, G, seed)
                if symm:
                    vs = load_vstate_vit_symm(f"{base}_symm.mpack", f"{base}_metadata.json", hi, g, patches, distances)
                else:
                    vs = load_vstate_vit(f"{base}.mpack", f"{base}_metadata.json", hi, patches, distances)
                vs.n_samples = n_samples
                res = {}
                for k, op in ops.items():
                    st = vs.expect(op)
                    res[k] = {"mean": float(np.real(st.mean)), "variance": float(np.real(st.variance)),
                              "error": float(np.real(st.error_of_mean))}
                out = f"{base}{'_symm' if symm else ''}_sz.json"
                json.dump(res, open(out, "w"), indent=2)
                print(f"{shape} G={G} seed={seed}: E={res['E']['mean']:.6f}  "
                      f"S^z={g.n_nodes * res['Mz']['mean'] / 2:+.4f}  -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compute", action="store_true", help="measure E, M_z for seeds without observables")
    p.add_argument("--symm", action="store_true", help="use the symmetry-projected states")
    p.add_argument("--gens", type=int, nargs="+", default=GENS)
    p.add_argument("--retrained", action="store_true",
                   help="before/after figure of the fixed-S^z retraining (img/retrained_<shape>.pdf)")
    args = p.parse_args()

    if args.retrained:
        for shape in SHAPES:
            figure_retrained(shape, gens=args.gens)
        raise SystemExit
    if args.compute:
        compute_missing(args.gens, args.symm)
    for shape in SHAPES:
        figure(shape, args.gens, args.symm)

# %%
