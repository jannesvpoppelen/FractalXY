import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.optimize import curve_fit


# Use matplotlib's math renderer instead of full LaTeX (looks the same, no LaTeX needed)
plt.rcParams.update({
    "text.usetex": False,  # Use matplotlib's mathtext instead
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "DejaVu Sans"],  # Use DejaVu Serif for better math rendering
    "mathtext.fontset": "cm",  # or "cm" for Computer Modern (LaTeX style)
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10
})

def plot_correlators(observables_files, labels=None, save_prefix="correlators"):
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(observables_files))]
    
    all_observables = []
    for filepath in observables_files:
        with open(filepath, 'r') as f:
            all_observables.append(json.load(f))
    
    graph_data = []  # List (d_list, c_list, c_err_list, label)
    corner_data = []
    euclidean_data = []
    
    for obs, label in zip(all_observables, labels):
        # Graph correlators
        d_list = []
        c_list = []
        c_err_list = []
        for key in obs.keys():
            if key.startswith("Cxy_graph_r"):
                d = int(key.split("Cxy_graph_r")[1])
                d_list.append(d)
                c_list.append(obs[key]["mean"])
                c_err_list.append(np.real(obs[key]["error"]))
        if len(d_list) > 0:
            graph_data.append((d_list, c_list, c_err_list, label))
        
        # Corner correlators
        d_corner_list = []
        c_corner_list = []
        c_corner_err_list = []
        for key in obs.keys():
            if key.startswith("Cxy_corner_d"):
                d = int(key.split("Cxy_corner_d")[1])
                d_corner_list.append(d)
                c_corner_list.append(obs[key]["mean"])
                c_corner_err_list.append(np.real(obs[key]["error"]))
        if len(d_corner_list) > 0:
            corner_data.append((d_corner_list, c_corner_list, c_corner_err_list, label))
        
        # Euclidean distance correlators
        d_euclidean_list = []
        c_euclidean_list = []
        c_euclidean_err_list = []
        for key in obs.keys():
            if key.startswith("Cxy_euclidean_r"):
                d = float(key.split("Cxy_euclidean_r")[1])
                d_euclidean_list.append(d)
                c_euclidean_list.append(obs[key]["mean"])
                c_euclidean_err_list.append(np.real(obs[key]["error"]))
        if len(d_euclidean_list) > 0:
            euclidean_data.append((d_euclidean_list, c_euclidean_list, c_euclidean_err_list, label))
    
    markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p']
    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # layout: 2x2 if euclidean data exists, otherwise 1x2
    has_euclidean = len(euclidean_data) > 0
    if has_euclidean:
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        ax_graph_lin = axs[0, 0]
        ax_graph_log = axs[0, 1]
        ax_eucl_lin = axs[1, 0]
        ax_eucl_log = axs[1, 1]
    else:
        fig, (ax_graph_lin, ax_graph_log) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax_graph_lin.set_xlabel(r'Graph Distance $r$', fontsize=18)
    ax_graph_lin.set_ylabel(r'$C_{xy}(r)$', fontsize=18)
    ax_graph_lin.grid(alpha=0.3)
    
    ax_graph_log.set_xlabel(r'Graph Distance $r$', fontsize=18)
    ax_graph_log.set_ylabel(r'$C_{xy}(r)$', fontsize=18)
    ax_graph_log.set_xscale('log')
    ax_graph_log.set_yscale('log')
    ax_graph_log.grid(alpha=0.3, which='both', linestyle='--')
    
    # Plot graph correlators
    for idx, (d_list, c_list, c_err_list, label) in enumerate(graph_data):
        marker = markers[idx % len(markers)]
        color = base_colors[idx % len(base_colors)]
        
        d_arr = np.array(d_list)
        c_arr = np.array(c_list)
        c_err_arr = np.array(c_err_list)
        order = np.argsort(d_arr)
        d_arr = d_arr[order]
        c_arr = c_arr[order]
        c_err_arr = c_err_arr[order]
        
        mask = c_arr > 0
        d_pos = d_arr[mask]
        c_pos = c_arr[mask]
        c_err_pos = c_err_arr[mask]
        
        ax_graph_lin.errorbar(d_arr, c_arr, yerr=c_err_arr, fmt=marker,
                             markersize=8, linewidth=2, markeredgewidth=2,
                             linestyle='-', capsize=4, color=color,
                             label=label, zorder=3)
        
        if len(d_pos) >= 2:
            # Power-law fit for log-log plot
            try:
                log_r = np.log(d_pos)
                log_c = np.log(c_pos)
                poly = np.polyfit(log_r, log_c, 1)
                eta = -poly[0]
                A = np.exp(poly[1])
                print(f"{label} (graph): η = {eta:.4f}, A = {A:.4f}")
                
                ax_graph_log.scatter(d_pos, c_pos, marker=marker, s=64, 
                             color=color, label=label, zorder=3)
                r_fine = np.logspace(np.log10(d_pos.min()), np.log10(d_pos.max()), 100)
                ax_graph_log.loglog(r_fine, A * r_fine**(-eta), '--', 
                            linewidth=2.5, color=color, alpha=0.7, zorder=2)
            except:
                print(f"{label} (graph): Power-law fit failed")
                ax_graph_log.scatter(d_pos, c_pos, marker=marker, s=64, color=color, 
                             label=label, zorder=3)
    
    # Plot corner correlators
    for idx, (d_list, c_list, c_err_list, label) in enumerate(corner_data):
        marker = markers[idx % len(markers)]
        color = base_colors[idx % len(base_colors)]
        
        d_arr = np.array(d_list)
        c_arr = np.array(c_list)
        c_err_arr = np.array(c_err_list)
        order = np.argsort(d_arr)
        d_arr = d_arr[order]
        c_arr = c_arr[order]
        c_err_arr = c_err_arr[order]
        
        mask = c_arr > 0
        d_pos = d_arr[mask]
        c_pos = c_arr[mask]
        c_err_pos = c_err_arr[mask]
        
        ax_graph_lin.errorbar(d_arr, c_arr, yerr=c_err_arr, fmt=marker,
                             markersize=8, linewidth=2, markeredgewidth=2,
                             linestyle='--', capsize=4, color=color,
                             markerfacecolor='none', markeredgecolor=color,
                             label=f'{label} (corner)', zorder=3, alpha=0.7)
        
        if len(d_pos) >= 2:
            # Power-law fit for log-log plot
            try:
                log_r = np.log(d_pos)
                log_c = np.log(c_pos)
                poly = np.polyfit(log_r, log_c, 1)
                eta = -poly[0]
                A = np.exp(poly[1])
                print(f"{label} (corner): η = {eta:.4f}, A = {A:.4f}")
                
                ax_graph_log.scatter(d_pos, c_pos, marker=marker, s=64, 
                             facecolors='none', edgecolors=color, linewidths=2,
                             zorder=3, alpha=0.7)
                r_fine = np.logspace(np.log10(d_pos.min()), np.log10(d_pos.max()), 100)
                ax_graph_log.loglog(r_fine, A * r_fine**(-eta), ':', 
                            linewidth=2, color=color, alpha=0.6, zorder=2)
            except:
                print(f"{label} (corner): Power-law fit failed")
                ax_graph_log.scatter(d_pos, c_pos, marker=marker, s=64,
                             facecolors='none', edgecolors=color, linewidths=2,
                             label=f'{label} (corner)', zorder=3, alpha=0.7)
    
    ax_graph_lin.text(0.08, 0.98, 'Filled marker: all-to-all correlators\nHollow marker: corner-to-corner correlators', 
                     transform=ax_graph_lin.transAxes, fontsize=10,
                     verticalalignment='top', bbox=dict(boxstyle='round', 
                     facecolor='wheat', alpha=0.8))
    ax_graph_log.text(0.02, 0.12, 'Filled marker: all-to-all correlators\nHollow marker: corner-to-corner correlators', 
                     transform=ax_graph_log.transAxes, fontsize=10,
                     verticalalignment='top', bbox=dict(boxstyle='round', 
                     facecolor='wheat', alpha=0.8))
    
    ax_graph_lin.legend(fontsize=11, framealpha=0.9, loc='best')
    ax_graph_log.legend(fontsize=11, framealpha=0.9, loc='best')
    
    # Euclidean distance correlators
    if has_euclidean:
        ax_eucl_lin.set_xlabel(r'Euclidean Distance $r$', fontsize=18)
        ax_eucl_lin.set_ylabel(r'$C_{xy}(r)$', fontsize=18)
        ax_eucl_lin.grid(alpha=0.3)
        
        ax_eucl_log.set_xlabel(r'Euclidean Distance $r$', fontsize=18)
        ax_eucl_log.set_ylabel(r'$C_{xy}(r)$', fontsize=18)
        ax_eucl_log.set_xscale('log')
        ax_eucl_log.set_yscale('log')
        ax_eucl_log.grid(alpha=0.3, which='both', linestyle='--')
        
        for idx, (d_list, c_list, c_err_list, label) in enumerate(euclidean_data):
            marker = markers[idx % len(markers)]
            color = base_colors[idx % len(base_colors)]
            
            d_arr = np.array(d_list)
            c_arr = np.array(c_list)
            c_err_arr = np.array(c_err_list)
            order = np.argsort(d_arr)
            d_arr = d_arr[order]
            c_arr = c_arr[order]
            c_err_arr = c_err_arr[order]
            
            mask = c_arr > 0
            d_pos = d_arr[mask]
            c_pos = c_arr[mask]
            c_err_pos = c_err_arr[mask]
            
            ax_eucl_lin.errorbar(d_arr, c_arr, yerr=c_err_arr, fmt=marker,
                                markersize=8, linewidth=2, markeredgewidth=2,
                                linestyle='-', capsize=4, color=color,
                                label=label, zorder=3)
            
            if len(d_pos) >= 2:
                # Power-law fit for log-log plot
                try:
                    log_r = np.log(d_pos)
                    log_c = np.log(c_pos)
                    poly = np.polyfit(log_r, log_c, 1)
                    eta = -poly[0]
                    A = np.exp(poly[1])
                    print(f"{label} (euclidean): η = {eta:.4f}, A = {A:.4f}")
                    
                    ax_eucl_log.scatter(d_pos, c_pos, marker=marker, s=64, 
                                 color=color, label=label, zorder=3)
                    r_fine = np.logspace(np.log10(d_pos.min()), np.log10(d_pos.max()), 100)
                    ax_eucl_log.loglog(r_fine, A * r_fine**(-eta), '--', 
                                linewidth=2.5, color=color, alpha=0.7, zorder=2)
                except:
                    print(f"{label} (euclidean): Power-law fit failed")
                    ax_eucl_log.scatter(d_pos, c_pos, marker=marker, s=64, color=color, 
                                 label=label, zorder=3)
        
        ax_eucl_lin.legend(fontsize=11, framealpha=0.9, loc='best')
        ax_eucl_log.legend(fontsize=11, framealpha=0.9, loc='best')
    
    plt.tight_layout()
    plt.savefig(f"{save_prefix}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nSaved figure to {save_prefix}.png")


def fidelity_figures():
    # Load fidelity results from JSON files and create heatmaps
    generations = [1, 2, 3, 4]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, gen in enumerate(generations):
        filename = f"data/triangular/fidelity_triangular_gasket_G={gen}.json"
        with open(filename, "r") as f:
            data = json.load(f)
        
        seeds = data["seeds"]
        n_seeds = len(seeds)
        
        # Build symmetric fidelity matrix
        fidelity_matrix = np.ones((n_seeds, n_seeds))  # Diagonal = 1 (self-fidelity)
        
        for key, value in data["fidelities"].items():
            # Parse seed indices from key like "seed1_seed2"
            parts = key.split("_")
            i = int(parts[0].replace("seed", "")) - 1  # Convert to 0-indexed
            j = int(parts[1].replace("seed", "")) - 1
            fidelity_matrix[i, j] = value
            fidelity_matrix[j, i] = value  # Symmetric
        
        # Create heatmap
        im = axes[idx].imshow(fidelity_matrix, cmap='gnuplot', vmin=0, vmax=1, 
                             aspect='auto', interpolation='nearest')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        cbar.set_label('Fidelity', fontsize=12)
        
        # Set ticks and labels
        axes[idx].set_xticks(range(n_seeds))
        axes[idx].set_yticks(range(n_seeds))
        axes[idx].set_xticklabels([f'S{s}' for s in seeds], fontsize=11)
        axes[idx].set_yticklabels([f'S{s}' for s in seeds], fontsize=11)
        
        # Add text annotations
        for i in range(n_seeds):
            for j in range(n_seeds):
                text = axes[idx].text(j, i, f'{fidelity_matrix[i, j]:.3f}',
                                     ha="center", va="center", color="white" if fidelity_matrix[i, j] < 0.5 else "black",
                                     fontsize=10)
        
        axes[idx].set_title(f'Generation {gen}\n(Mean: {data["mean_fidelity"]:.4f}, Std: {data["std_fidelity"]:.4f})', 
                           fontsize=13, pad=10)
        axes[idx].set_xlabel('Seed', fontsize=12)
        axes[idx].set_ylabel('Seed', fontsize=12)
    
    plt.tight_layout()
    plt.savefig("fidelity_heatmaps.png", dpi=300, bbox_inches='tight')
    plt.show()

# fidelity_figures()



plot_correlators(
    observables_files=[
        "data/RBM/triangular/triangular_gasket_G=0_seed=1_observables.json",
        "data/RBM/triangular/triangular_gasket_G=1_seed=1_observables.json",
        "data/ViT/triangular/vit_triangular_G=3_seed=1_observables.json",
        "data/ViT/triangular/vit_triangular_G=4_seed=1_observables.json"],
    labels=["Gen 1", "Gen 2", "Gen 3", "Gen 4"],
    save_prefix="triangle_xy_correlator_analysis"
)


plot_correlators(
    observables_files=[
        "data/RBM/honeycomb/honeycomb_gasket_G=0_seed=1_observables.json",
        "data/RBM/honeycomb/honeycomb_gasket_G=1_seed=1_observables.json",
        "data/ViT/honeycomb/vit_honeycomb_G=3_seed=1_observables.json",
        "data/ViT/honeycomb/vit_honeycomb_G=4_seed=1_observables.json"],
    labels=["Gen 1", "Gen 2", "Gen 3", "Gen 4"],
    save_prefix="honeycomb_xy_correlator_analysis"
)



plot_correlators(
    observables_files=[
        "data/ViT/triangular/vit_triangular_G=3_seed=1_observables.json",
        "data/ViT/triangular/vit_triangular_G=3_seed=1_symm_observables.json",
        "data/ViT/triangular/vit_triangular_G=4_seed=1_observables.json",
        "data/ViT/triangular/vit_triangular_G=4_seed=1_symm_observables.json"],
    labels=["Gen 3", "Gen 3 symmetry-projected", "Gen 4", "Gen 4 symmetry-projected"],
    save_prefix="triangle_xy_symm_correlator_analysis"
)

plot_correlators(
    observables_files=[
        "data/ViT/honeycomb/vit_honeycomb_G=3_seed=1_observables.json",
        "data/ViT/honeycomb/vit_honeycomb_G=3_seed=1_symm_observables.json"],
    labels=["Gen 3", "Gen 3 symmetry-projected"],
    save_prefix="honeycomb_xy_symm_correlator_analysis"
)