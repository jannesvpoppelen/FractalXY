import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from fractals import honeycomb_gasket, triangular_gasket

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except ImportError:
    pass
plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif"})

PAGE = 6.75
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#999999"]
STYLES = ["-", "--", ":", "-."]


def n_sites(shape, G, n=2):
    build = triangular_gasket if shape == "triangular" else honeycomb_gasket
    return build(G, n)[0].shape[0]


def load_history(run, shape, G, seed):
    """(energies, variances, {stage: first step}, source) of one seed, or None."""
    base = f"{run}/{shape}/vit_{shape}_G={G}_seed={seed}"
    for path in (f"{base}_symm_energy_history.json", f"{base}_energy_history.json"):
        if os.path.exists(path):
            h = json.load(open(path))
            starts = {}
            if "attn_start_iter" in h and h["symm_start_iter"] > h["attn_start_iter"]:
                starts["attn"] = h["attn_start_iter"]
            if h.get("symm_energies"):
                starts["c3v"] = h["symm_start_iter"]
            e = h["energies"] + h.get("symm_energies", [])
            v = h["variances"] + h.get("symm_variances", [])
            return np.array(e), np.array(v), starts, path

    path = f"{base}_ckpt.json"
    if os.path.exists(path):
        ck = json.load(open(path))
        hist = ck.get("history") or {   # checkpoints from before the symmetric-attention stage
            "main": {"energies": ck["energies"], "variances": ck["variances"]},
            "c3v": {"energies": ck["symm_energies"], "variances": ck["symm_variances"]},
        }
        e, v, starts = [], [], {}
        for stage in ("main", "attn", "c3v"):
            if stage not in hist or not hist[stage]["energies"]:
                continue
            if stage != "main":
                starts[stage] = len(e)
            e += hist[stage]["energies"]
            v += hist[stage]["variances"]
        return np.array(e), np.array(v), starts, f"{path} (still running)"
    return None


def running_mean(y, k):
    if k <= 1 or len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="valid")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shape", required=True, choices=["triangular", "honeycomb"])
    p.add_argument("--gen", type=int, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--runs", nargs="+", default=["data/ViT_sz"], help="output directories of trainViT_sz.py")
    p.add_argument("--labels", nargs="+", help="legend names of the runs (default: directory names)")
    p.add_argument("--ref", type=float, help="reference energy (default: data/ED/ if available)")
    p.add_argument("--smooth", type=int, default=1, help="running mean over this many steps")
    p.add_argument("--out", help="output file (default img/training_<shape>_G=<gen>.pdf)")
    args = p.parse_args()

    N = n_sites(args.shape, args.gen)
    s0 = 0.0 if N % 2 == 0 else 0.5
    e_ref = args.ref
    ed_file = f"data/ED/{args.shape}_G={args.gen}_sz={s0:+g}.json"
    if e_ref is None and os.path.exists(ed_file):
        e_ref = json.load(open(ed_file))["energies"][0]
    print(f"{args.shape} G={args.gen}: N={N}, reference energy "
          + (f"{e_ref:.10f}" if e_ref is not None else "none (plotting E itself)"))

    labels = args.labels or [os.path.basename(os.path.normpath(r)) for r in args.runs]
    fig, (ax_e, ax_v) = plt.subplots(1, 2, figsize=(PAGE, 2.7))
    shown = []
    for r, (run, label) in enumerate(zip(args.runs, labels)):
        for k, seed in enumerate(args.seeds):
            h = load_history(run, args.shape, args.gen, seed)
            if h is None:
                print(f"  {label} seed {seed}: no history in {run}")
                continue
            e, v, starts, source = h
            y = e - e_ref if e_ref is not None else e
            ys, vs = running_mean(y, args.smooth), running_mean(v, args.smooth)
            x = np.arange(len(ys)) + (args.smooth - 1 if len(y) >= args.smooth > 1 else 0)
            style = dict(color=COLORS[r % len(COLORS)], ls=STYLES[k % len(STYLES)], lw=1.0)
            ax_e.plot(x, ys, label=f"{label}, seed {seed}", **style)
            ax_v.plot(x, np.maximum(vs, 1e-16), **style)
            for step in starts.values():
                for a in (ax_e, ax_v):
                    a.axvline(step, color=style["color"], ls=":", lw=0.7, alpha=0.7)
            shown.append(ys)
            tail = slice(max(0, len(y) - 50), None)
            print(f"  {label} seed {seed}: {len(e)} steps ({source}); last 50 steps: "
                  f"{'E - E_ref' if e_ref is not None else 'E'} = {np.mean(y[tail]):+.3e}, "
                  f"var = {np.mean(v[tail]):.2e}")
    if not shown:
        raise SystemExit("nothing to plot")

    if e_ref is not None:
        allv = np.concatenate(shown)
        if (allv <= 0).any():
            # Monte Carlo noise around the exact energy: linear below the noise level
            lin = 10 ** np.ceil(np.log10(np.abs(allv[allv <= 0]).max() + 1e-300))
            ax_e.set_yscale("symlog", linthresh=lin)
            ax_e.axhspan(-lin, lin, color="0.92", zorder=0)
        else:
            ax_e.set_yscale("log")
        ax_e.set_ylabel(r"$E - E_{\rm exact}$")
    else:
        ax_e.set_ylabel(r"$E$")
    ax_v.set_yscale("log")
    ax_v.set_ylabel(r"Var$(H)$")
    for a in (ax_e, ax_v):
        a.set_xlabel("step")
    ax_e.legend(fontsize=6, frameon=False)
    fig.suptitle(f"{args.shape.capitalize()} gasket $G={args.gen}$ ($N={N}$, $S^z={s0:g}$)"
                 + (f", running mean over {args.smooth}" if args.smooth > 1 else ""), fontsize=9)
    fig.tight_layout(pad=0.3)
    out = args.out or f"img/training_{args.shape}_G={args.gen}.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
