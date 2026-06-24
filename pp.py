import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as mpe
import numpy as np
import json
import networkx as nx
import scienceplots

from fractals import triangular_gasket, honeycomb_gasket, honeycomb_gasket_coordinates


PRB_SINGLE = 3.375
PRB_DOUBLE = 6.75

COLORS = [
    "#6BAED6", "#74C476", "#9E9AC8",
    "#FB6A4A", "#41B6C4", "#969696", "#BCBDDC",
]
MARKERS = ["o", "s", "^", "D", "v", "p", "<"]


def _load_correlators(observables_files, labels):
    graph_data = []
    euclidean_data = []

    for filepath, label in zip(observables_files, labels):
        with open(filepath, "r") as f:
            obs = json.load(f)

        graph = {
            int(k.split("Cxy_graph_r")[1]): obs[k]
            for k in obs if k.startswith("Cxy_graph_r")
        }
        if graph:
            d = np.array(sorted(graph))
            c = np.array([graph[r]["mean"] for r in d])
            e = np.array([np.real(graph[r]["error"]) for r in d])
            graph_data.append((d, c, e, label))

        eucl = {
            float(k.split("Cxy_euclidean_r")[1]): obs[k]
            for k in obs if k.startswith("Cxy_euclidean_r")
        }
        if eucl:
            d = np.array(sorted(eucl))
            c = np.array([eucl[r]["mean"] for r in d])
            e = np.array([np.real(eucl[r]["error"]) for r in d])
            euclidean_data.append((d, c, e, label))

    return graph_data, euclidean_data


def _power_law_fit(d_pos, c_pos):
    poly = np.polyfit(np.log(d_pos), np.log(c_pos), 1)
    return -poly[0], np.exp(poly[1])


def _add_fractal_inset(ax, lattice_type, g=2, n=3):
    inset = ax.inset_axes([0.55, 0.55, 0.42, 0.42])
    for spine in inset.spines.values():
        spine.set_visible(False)
    inset.patch.set_alpha(0.0)
    inset.set_facecolor("none")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_aspect("equal")

    if lattice_type == "triangular":
        adj, _ = triangular_gasket(g, n)
        xa, ya, _, _ = honeycomb_gasket_coordinates(g, n)
        graph = nx.from_numpy_array(adj.toarray())
        pos = dict(enumerate(zip(xa, ya)))
    else:
        xa, ya, xb, yb = honeycomb_gasket_coordinates(g, n)
        x = np.hstack((xa, xb))
        y = np.hstack((ya, yb))
        adj, _ = honeycomb_gasket(g, n)
        graph = nx.from_numpy_array(adj.toarray())
        pos = dict(enumerate(zip(x, y)))

    nx.draw(graph, pos=pos, ax=inset, node_size=0, edge_color="black", width=0.6)


def _pow10_fmt(val, pos):
    """Label only exact powers of 10; skip everything else."""
    if val <= 0:
        return ""
    e = round(np.log10(val))
    if not np.isclose(val, 10.0 ** e, rtol=1e-6):
        return ""
    if e == 0:
        return r"$1$"
    if e == 1:
        return r"$10$"
    return r"$10^{%d}$" % e


def _plot_panel(ax, datasets, xlabel, show_legend=False, label_text=None,
                ylim=None, yticks=None, xlim=None, xticks=None):
    from matplotlib.lines import Line2D

    # Draw in reverse so smaller (earlier) datasets sit on top
    for idx in reversed(range(len(datasets))):
        d_arr, c_arr, c_err_arr, label = datasets[idx]
        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]

        mask = c_arr > 0
        d_pos = d_arr[mask]
        c_pos = c_arr[mask]

        ax.errorbar(
            d_arr, c_arr, yerr=c_err_arr,
            fmt=marker, color=color,
            markersize=4.5, linewidth=0.8, markeredgewidth=0.8,
            capsize=2, capthick=1, linestyle="none", zorder=1,
        )

        outline=mpe.withStroke(linewidth=0.6, foreground='black')

        if len(d_pos) >= 2:
            try:
                eta, A = _power_law_fit(d_pos, c_pos)
                r_fine = np.logspace(np.log10(d_pos.min()), np.log10(d_pos.max()), 200)
                ax.plot(
                    r_fine, A * r_fine ** (-eta),
                    color=color, linewidth=1.2, linestyle="--", alpha=0.85, zorder=3, path_effects=[outline]
                )
                print(f"  {label}: eta = {eta:.4f},  A = {A:.4f}")
            except Exception:
                print(f"  {label}: power-law fit failed")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(r"$C_{xy}$", fontsize=20)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)

    if xlim is not None:
        ax.set_xlim(xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
        # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, p: f"${v:g}$"))
        # ax.xaxis.set_minor_locator(ticker.NullLocator())

    if ylim is not None:
        ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, p: f"${v:g}$"))
        ax.yaxis.set_minor_locator(ticker.NullLocator())
    else:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_pow10_fmt))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_pow10_fmt))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=16)
    ax.tick_params(axis="both", which="major", length=4)
    ax.tick_params(axis="both", which="minor", length=2)

    if show_legend:
        handles = [
            Line2D([0], [0], color=COLORS[i % len(COLORS)],
                   marker=MARKERS[i % len(MARKERS)], linestyle="None",
                   markersize=4, label=datasets[i][3])
            for i in range(len(datasets))
        ]
        ax.legend(handles=handles, fontsize=12, frameon=True, framealpha=0.9,
                  edgecolor="0.5", handlelength=1.2, borderpad=0.5,
                  loc="lower left")

    if label_text:
        ax.text(0.88, 0.94, label_text, transform=ax.transAxes,
                fontsize=16, va="top", ha="left", zorder=10,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.8))


def plot_correlators_loglog(
    triangular_files,
    triangular_labels,
    honeycomb_files,
    honeycomb_labels,
    save_prefix="correlators_loglog",
    width=PRB_DOUBLE,
    inset_generation=2,
    xlim_tri=(0, 18),
    xticks_tri=(0.7, 18),
    ylim_tri=(0.4, 1.5),
    yticks_tri=(0.5, 1.),
    xlim_hc=(0.4, 40),
    xticks_hc=(0.4, 40),
    ylim_hc=(0.3, 1.6),
    yticks_hc=(0.5, 1.0),
):
    tri_graph, tri_eucl = _load_correlators(triangular_files, triangular_labels)
    hc_graph,  hc_eucl  = _load_correlators(honeycomb_files,  honeycomb_labels)

    has_euclidean = len(tri_eucl) > 0 or len(hc_eucl) > 0
    n_rows = 2 if has_euclidean else 1
    height = width * (0.4 * n_rows)

    panel_labels = [["(a)", "(b)"], ["(c)", "(d)"]]

    with plt.style.context(["science", "no-latex"]):
        plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif"})

        fig, axes = plt.subplots(n_rows, 2, figsize=(width, height), squeeze=False, sharex="col")

        print("--- Network distance ---")
        _plot_panel(axes[0, 0], tri_graph, r"$d$",
                    show_legend=True, label_text=panel_labels[0][0],
                    xlim=xlim_tri, xticks=xticks_tri,
                    ylim=ylim_tri, yticks=yticks_tri)
        _plot_panel(axes[0, 1], hc_graph,  r"$d$",
                    show_legend=False, label_text=panel_labels[0][1],
                    xlim=xlim_hc, xticks=xticks_hc,
                    ylim=ylim_hc, yticks=yticks_hc)

        _add_fractal_inset(axes[0, 0], "triangular",  g=inset_generation)
        _add_fractal_inset(axes[0, 1], "honeycomb",   g=inset_generation)

        if has_euclidean:
            print("--- Euclidean distance ---")
            _plot_panel(axes[1, 0], tri_eucl, r"$r$",
                        show_legend=False, label_text=panel_labels[1][0],
                        ylim=ylim_tri, yticks=yticks_tri)
            _plot_panel(axes[1, 1], hc_eucl,  r"$r$",
                        show_legend=False, label_text=panel_labels[1][1],
                        ylim=ylim_hc, yticks=yticks_hc)

        for ax in [axes[0, 1]] + ([axes[1, 1]] if has_euclidean else []):
            ax.set_ylabel("")

        fig.tight_layout(pad=0.3)
        fig.subplots_adjust(wspace=0.15)
        fig.savefig(f"{save_prefix}.pdf", dpi=600, bbox_inches="tight")
        fig.savefig(f"{save_prefix}.png", dpi=600, bbox_inches="tight")
        print(f"\nSaved: {save_prefix}.pdf  &  {save_prefix}.png")
        plt.show()


if __name__ == "__main__":
    plot_correlators_loglog(
        triangular_files=[
            "data/RBM/triangular/triangular_gasket_G=0_seed=1_observables.json",
            "data/RBM/triangular/triangular_gasket_G=1_seed=1_observables.json",
            "data/ViT/triangular/vit_triangular_G=3_seed=1_observables.json",
            "data/ViT/triangular/vit_triangular_G=4_seed=1_observables.json",
        ],
        triangular_labels=["$G=1$", "$G=2$", "$G=3$", "$G=4$"],
        honeycomb_files=[
            "data/RBM/honeycomb/honeycomb_gasket_G=0_seed=1_observables.json",
            "data/RBM/honeycomb/honeycomb_gasket_G=1_seed=1_observables.json",
            "data/ViT/honeycomb/vit_honeycomb_G=3_seed=1_observables.json",
            "data/ViT/honeycomb/vit_honeycomb_G=4_seed=1_observables.json",
        ],
        honeycomb_labels=["$G=1$", "$G=2$", "$G=3$", "$G=4$"],
        save_prefix="correlators",
        inset_generation=2,
    )
