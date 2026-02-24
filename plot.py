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


# Euclidean distance correlators
# d1_euclidean_list = []
# c1_euclidean_list = []
# c1_euclidean_err_list = []
# for key in observables1.keys():
#     if key.startswith("Cxy_euclidean_r"):
#         d = float(key.split("Cxy_euclidean_r")[1])
#         d1_euclidean_list.append(d)
#         c1_euclidean_list.append(observables1[key]["mean"])
#         c1_euclidean_err_list.append(np.sqrt(observables1[key]["variance"]))

# d_arr = np.array(d1_euclidean_list)
# c_arr = np.array(c1_euclidean_list)
# order = np.argsort(d_arr)
# d_arr = d_arr[order]
# c_arr = c_arr[order]


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
                c_err_list.append(np.sqrt(obs[key]["variance"]))
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
                c_corner_err_list.append(np.sqrt(obs[key]["variance"]))
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
                c_euclidean_err_list.append(np.sqrt(obs[key]["variance"]))
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
    
    ax_graph_lin.text(0.02, 0.98, 'Filled marker: all-to-all correlators\nHollow marker: corner-to-corner correlators', 
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


def plot_correlators_averaged(observables_by_generation, labels=None, save_prefix="correlators_averaged"):
    """
    Plot correlators averaged across multiple seeds for each generation.
    
    Parameters:
    -----------
    observables_by_generation : dict or list
        If dict: keys are generation labels, values are lists of observable file paths (one per seed)
        If list: list of lists, where each sublist contains observable files for that generation
        Example: {"Gen 1": ["gen1_seed1.json", "gen1_seed2.json"], 
                  "Gen 2": ["gen2_seed1.json", "gen2_seed2.json"]}
    labels : list, optional
        Custom labels for each generation. If None, uses dict keys or "Gen 1", "Gen 2", etc.
    save_prefix : str
        Prefix for saved figure filename
    """
    # Convert to dict format if list is provided
    if isinstance(observables_by_generation, list):
        observables_by_generation = {f"Gen {i+1}": files 
                                     for i, files in enumerate(observables_by_generation)}
    
    if labels is None:
        labels = list(observables_by_generation.keys())
    else:
        # If custom labels provided, create new dict with those labels
        observables_by_generation = {label: files 
                                     for label, files in zip(labels, observables_by_generation.values())}
    
    graph_data = []  # List of (d_list, c_mean_list, c_sem_list, label)
    corner_data = []
    euclidean_data = []
    
    for gen_label in labels:
        files = observables_by_generation[gen_label]
        n_seeds = len(files)
        
        # Load all observables for this generation
        all_obs = []
        for filepath in files:
            with open(filepath, 'r') as f:
                all_obs.append(json.load(f))
        
        # --- Graph correlators ---
        # Collect all unique distances
        distances = set()
        for obs in all_obs:
            for key in obs.keys():
                if key.startswith("Cxy_graph_r"):
                    d = int(key.split("Cxy_graph_r")[1])
                    distances.add(d)
        
        if len(distances) > 0:
            distances = sorted(distances)
            c_means = []
            c_sems = []
            
            for d in distances:
                key = f"Cxy_graph_r{d}"
                values = []
                for obs in all_obs:
                    if key in obs:
                        values.append(obs[key]["mean"])
                
                if len(values) > 0:
                    c_mean = np.mean(values)
                    c_sem = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
                    c_means.append(c_mean)
                    c_sems.append(c_sem)
                else:
                    c_means.append(np.nan)
                    c_sems.append(np.nan)
            
            graph_data.append((distances, c_means, c_sems, gen_label))
        
        # --- Corner correlators ---
        distances_corner = set()
        for obs in all_obs:
            for key in obs.keys():
                if key.startswith("Cxy_corner_d"):
                    d = int(key.split("Cxy_corner_d")[1])
                    distances_corner.add(d)
        
        if len(distances_corner) > 0:
            distances_corner = sorted(distances_corner)
            c_corner_means = []
            c_corner_sems = []
            
            for d in distances_corner:
                key = f"Cxy_corner_d{d}"
                values = []
                for obs in all_obs:
                    if key in obs:
                        values.append(obs[key]["mean"])
                
                if len(values) > 0:
                    c_mean = np.mean(values)
                    c_sem = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
                    c_corner_means.append(c_mean)
                    c_corner_sems.append(c_sem)
                else:
                    c_corner_means.append(np.nan)
                    c_corner_sems.append(np.nan)
            
            corner_data.append((distances_corner, c_corner_means, c_corner_sems, gen_label))
        
        # --- Euclidean correlators ---
        distances_euclidean = set()
        for obs in all_obs:
            for key in obs.keys():
                if key.startswith("Cxy_euclidean_r"):
                    d = float(key.split("Cxy_euclidean_r")[1])
                    distances_euclidean.add(d)
        
        if len(distances_euclidean) > 0:
            distances_euclidean = sorted(distances_euclidean)
            c_euclidean_means = []
            c_euclidean_sems = []
            
            for d in distances_euclidean:
                key = f"Cxy_euclidean_r{d:.4f}"
                values = []
                for obs in all_obs:
                    # Match key with tolerance for floating point
                    for obs_key in obs.keys():
                        if obs_key.startswith("Cxy_euclidean_r"):
                            d_obs = float(obs_key.split("Cxy_euclidean_r")[1])
                            if abs(d_obs - d) < 1e-6:
                                values.append(obs[obs_key]["mean"])
                                break
                
                if len(values) > 0:
                    c_mean = np.mean(values)
                    c_sem = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
                    c_euclidean_means.append(c_mean)
                    c_euclidean_sems.append(c_sem)
                else:
                    c_euclidean_means.append(np.nan)
                    c_euclidean_sems.append(np.nan)
            
            euclidean_data.append((distances_euclidean, c_euclidean_means, c_euclidean_sems, gen_label))
    
    # Plotting setup
    markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p']
    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
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
    for idx, (d_list, c_list, c_sem_list, label) in enumerate(graph_data):
        marker = markers[idx % len(markers)]
        color = base_colors[idx % len(base_colors)]
        
        d_arr = np.array(d_list)
        c_arr = np.array(c_list)
        c_sem_arr = np.array(c_sem_list)
        
        # Remove NaN values
        mask = ~np.isnan(c_arr)
        d_arr = d_arr[mask]
        c_arr = c_arr[mask]
        c_sem_arr = c_sem_arr[mask]
        
        ax_graph_lin.errorbar(d_arr, c_arr, yerr=c_sem_arr, fmt=marker,
                             markersize=8, linewidth=2, markeredgewidth=2,
                             linestyle='-', capsize=4, color=color,
                             label=label, zorder=3)
        
        # For log plot, only use positive values
        mask_pos = c_arr > 0
        d_pos = d_arr[mask_pos]
        c_pos = c_arr[mask_pos]
        c_sem_pos = c_sem_arr[mask_pos]
        
        if len(d_pos) >= 2:
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
    for idx, (d_list, c_list, c_sem_list, label) in enumerate(corner_data):
        marker = markers[idx % len(markers)]
        color = base_colors[idx % len(base_colors)]
        
        d_arr = np.array(d_list)
        c_arr = np.array(c_list)
        c_sem_arr = np.array(c_sem_list)
        
        # Remove NaN values
        mask = ~np.isnan(c_arr)
        d_arr = d_arr[mask]
        c_arr = c_arr[mask]
        c_sem_arr = c_sem_arr[mask]
        
        ax_graph_lin.errorbar(d_arr, c_arr, yerr=c_sem_arr, fmt=marker,
                             markersize=8, linewidth=2, markeredgewidth=2,
                             linestyle='--', capsize=4, color=color,
                             markerfacecolor='none', markeredgecolor=color,
                             label=f'{label} (corner)', zorder=3, alpha=0.7)
        
        # For log plot, only use positive values
        mask_pos = c_arr > 0
        d_pos = d_arr[mask_pos]
        c_pos = c_arr[mask_pos]
        c_sem_pos = c_sem_arr[mask_pos]
        
        if len(d_pos) >= 2:
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
    
    ax_graph_lin.text(0.02, 0.98, 'Filled marker: all-to-all correlators\nHollow marker: corner-to-corner correlators', 
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
        
        for idx, (d_list, c_list, c_sem_list, label) in enumerate(euclidean_data):
            marker = markers[idx % len(markers)]
            color = base_colors[idx % len(base_colors)]
            
            d_arr = np.array(d_list)
            c_arr = np.array(c_list)
            c_sem_arr = np.array(c_sem_list)
            
            # Remove NaN values
            mask = ~np.isnan(c_arr)
            d_arr = d_arr[mask]
            c_arr = c_arr[mask]
            c_sem_arr = c_sem_arr[mask]
            
            ax_eucl_lin.errorbar(d_arr, c_arr, yerr=c_sem_arr, fmt=marker,
                                markersize=8, linewidth=2, markeredgewidth=2,
                                linestyle='-', capsize=4, color=color,
                                label=label, zorder=3)
            
            # For log plot, only use positive values
            mask_pos = c_arr > 0
            d_pos = d_arr[mask_pos]
            c_pos = c_arr[mask_pos]
            c_sem_pos = c_sem_arr[mask_pos]
            
            if len(d_pos) >= 2:
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
    
    print(f"\nSaved averaged correlators figure to {save_prefix}.png")


def fidelity_figures():
    # Load fidelity results from JSON files and create heatmaps
    generations = [1, 2, 3, 4]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, gen in enumerate(generations):
        filename = f"data/fidelity_gen{gen}.json"
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

fidelity_figures()


# Example usage of plot_correlators:
plot_correlators(
    observables_files=[
        "data/sierpinski_gen1_seed1_observables.json",
        "data/sierpinski_gen2_seed1_observables.json",
        "data/sierpinski_gen3_seed1_observables.json",
        "data/sierpinski_gen4_seed1_observables.json"],
    labels=["Gen 1", "Gen 2", "Gen 3", "Gen 4"],
    save_prefix="xy_correlator_analysis"
)


# Example usage of plot_correlators_averaged (averaging across multiple seeds):
plot_correlators_averaged(
    observables_by_generation={
        "Gen 1": ["data/sierpinski_gen1_seed1_observables.json",
                  "data/sierpinski_gen1_seed2_observables.json",
                  "data/sierpinski_gen1_seed3_observables.json",
                  "data/sierpinski_gen1_seed4_observables.json",
                   "data/sierpinski_gen1_seed5_observables.json"],
        "Gen 2": ["data/sierpinski_gen2_seed1_observables.json",
                  "data/sierpinski_gen2_seed2_observables.json",
                  "data/sierpinski_gen2_seed3_observables.json",
                  "data/sierpinski_gen2_seed4_observables.json",
                   "data/sierpinski_gen2_seed5_observables.json"],
        "Gen 3": ["data/sierpinski_gen3_seed1_observables.json",
                  "data/sierpinski_gen3_seed2_observables.json",
                  "data/sierpinski_gen3_seed3_observables.json",
                  "data/sierpinski_gen3_seed4_observables.json",
                   "data/sierpinski_gen3_seed5_observables.json"],
        "Gen 4": ["data/sierpinski_gen4_seed1_observables.json",
                  "data/sierpinski_gen4_seed2_observables.json",
                  "data/sierpinski_gen4_seed3_observables.json",
                  "data/sierpinski_gen4_seed4_observables.json",
                   "data/sierpinski_gen4_seed5_observables.json"]
    },
    save_prefix="xy_correlator_averaged"
)


###################################
# full pagewidth figure, outdated


# def create_figure():

    # # Load graph structures

    # edges1 = np.genfromtxt("edges1.txt", dtype=int)
    # edges_list1 = [tuple(map(int, e)) for e in edges1]
    # vertices1 = np.genfromtxt("vertices1.txt")
    # N1 = len(vertices1)
    # convergence1 = json.load(open("sierpinski_gen1_seed1.log"))
    # observables1 = json.load(open("sierpinski_gen1_seed1_observables.json"))


    # edges2 = np.genfromtxt("edges2.txt", dtype=int)
    # edges_list2 = [tuple(map(int, e)) for e in edges2]
    # vertices2 = np.genfromtxt("vertices2.txt")
    # N2 = len(vertices2)
    # convergence2 = json.load(open("sierpinski_gen2_seed1.log"))
    # observables2 = json.load(open("sierpinski_gen2_seed1_observables.json"))


    # edges3 = np.genfromtxt("edges3.txt", dtype=int)
    # edges_list3 = [tuple(map(int, e)) for e in edges3]
    # vertices3 = np.genfromtxt("vertices3.txt")
    # N3 = len(vertices3)
    # convergence3 = json.load(open("sierpinski_gen3_seed1.log"))
    # observables3 = json.load(open("sierpinski_gen3_seed1_observables.json"))


    # edges4 = np.genfromtxt("edges4.txt", dtype=int)
    # edges_list4 = [tuple(map(int, e)) for e in edges4]
    # vertices4 = np.genfromtxt("vertices4.txt")
    # N4 = len(vertices4)
    # convergence4 = json.load(open("sierpinski_gen4_seed1.log"))
    # observables4 = json.load(open("sierpinski_gen4_seed1_observables.json"))




#     # Create 2x3 subplot figure
#     fig, axs = plt.subplots(2, 3, figsize=(15, 10))
#     axs = axs.flatten()
    

#     xlabels = [r"$Iterations$", r"", r"$r$", r"$Iterations$", r"", r"$r$"]
#     ylabels = [r"$E$", r"", r"$C_{xy}$", r"", r"", r"$C_{xy}$"]

#     for i in range(6):
#         axs[i].set_xlabel(xlabels[i])
#         axs[i].set_ylabel(ylabels[i])

#     for ax in axs:
#         ax.xaxis.label.set_size(18)
#         ax.yaxis.label.set_size(18)

#     axs[1].set_xticks([])
#     axs[1].set_yticks([])
#     axs[4].set_xticks([])
#     axs[4].set_yticks([])
#     axs[1].set_xticklabels([])
#     axs[1].set_yticklabels([])
#     axs[4].set_xticklabels([])
#     axs[4].set_yticklabels([])

#     axs[5].set_xscale('log')
#     axs[5].set_yscale('log')

#     #-------------------------------------
#     # Site resolved magnetization^2 plots
#     #-------------------------------------

#     # Generate random test magnetization data
#     np.random.seed(42)  # For reproducibility

#     mx3 = np.array([observables3[f"Mx_{i}"]["mean"] for i in range(N3)])
#     mx4 = np.array([observables4[f"Mx_{i}"]["mean"] for i in range(N4)])
#     m2_perp_3 = np.array([observables3[f"M2_perp_{i}"]["mean"] for i in range(N3)])
#     m2_perp_4 = np.array([observables4[f"M2_perp_{i}"]["mean"] for i in range(N4)])

#     e3_variance = observables3["E"]["variance"]
#     e4_variance = observables4["E"]["variance"]


#     x3, y3 = vertices3[:, 0], vertices3[:, 1]
#     for i, j in edges_list3:
#         axs[1].plot([x3[i], x3[j]], [y3[i], y3[j]], 'k-', alpha=0.3, linewidth=0.8)
#     scatter3 = axs[1].scatter(x3, y3, c=m2_perp_3, s=30, zorder=5, 
#                               edgecolors='black', linewidth=0.5,
#                               cmap='viridis')
#     axs[1].set_aspect('equal')
#     axs[1].set_title(f"Gen 3: E variance = {e3_variance:.2e}", fontsize=14)
#     axs[1].grid(False)
    
#     cbar2 = fig.colorbar(scatter3, ax=axs[1], orientation='horizontal', 
#                        location='bottom', pad=0.05, fraction=0.046, aspect=20)
#     # cbar2.set_label(r'$\langle \sigma^x_i \rangle$', fontsize=14)
#     cbar2.set_label(r'$\langle M^2_{\perp,i} \rangle$', fontsize=14)

#     x4, y4 = vertices4[:, 0], vertices4[:, 1]
#     for i, j in edges_list4:
#         axs[4].plot([x4[i], x4[j]], [y4[i], y4[j]], 'k-', alpha=0.3, linewidth=0.8)
#     scatter4 = axs[4].scatter(x4, y4, c=m2_perp_4, s=30, zorder=5, 
#                               edgecolors='black', linewidth=0.5,
#                               cmap='viridis')
#     axs[4].set_aspect('equal')
#     axs[4].set_title(f"Gen 4: E variance = {e4_variance:.2e}", fontsize=14)
#     axs[4].grid(False)
    
#     cbar = fig.colorbar(scatter4, ax=axs[4], orientation='horizontal', 
#                        location='bottom', pad=0.05, fraction=0.046, aspect=20)
#     # cbar.set_label(r'$\langle \sigma^x_i \rangle$', fontsize=14)
#     cbar.set_label(r'$\langle M^2_{\perp,i} \rangle$', fontsize=14)

#     #-------------------------------------
#     # Energy and Mag. convergence plots
#     #-------------------------------------

#     iters1 = convergence1["Energy"]["iters"]
#     iters2 = convergence2["Energy"]["iters"]
#     iters3 = convergence3["Energy"]["iters"]
#     iters4 = convergence4["Energy"]["iters"]

#     energy1 = np.array(convergence1["Energy"]["Mean"]["real"])#/N1
#     mx_mean1 = np.array(convergence1["Mx"]["Mean"]["real"])
#     my_mean1 = np.array(convergence1["My"]["Mean"]["real"])
#     mz_mean1 = np.array(convergence1["Mz"]["Mean"]["real"])
#     M2_perp1 = np.array(convergence1["M2_perp"]["Mean"]["real"])

#     energy2 = np.array(convergence2["Energy"]["Mean"]["real"])#/N2
#     mx_mean2 = np.array(convergence2["Mx"]["Mean"]["real"])
#     my_mean2 = np.array(convergence2["My"]["Mean"]["real"])
#     mz_mean2 = np.array(convergence2["Mz"]["Mean"]["real"])
#     M2_perp2 = np.array(convergence2["M2_perp"]["Mean"]["real"])

#     energy3 = np.array(convergence3["Energy"]["Mean"]["real"])#/N3
#     mx_mean3 = np.array(convergence3["Mx"]["Mean"]["real"])
#     my_mean3 = np.array(convergence3["My"]["Mean"]["real"])
#     mz_mean3 = np.array(convergence3["Mz"]["Mean"]["real"])
#     M2_perp3 = np.array(convergence3["M2_perp"]["Mean"]["real"])

#     energy4 = np.array(convergence4["Energy"]["Mean"]["real"])#/N4
#     mx_mean4 = np.array(convergence4["Mx"]["Mean"]["real"])
#     my_mean4 = np.array(convergence4["My"]["Mean"]["real"])
#     mz_mean4 = np.array(convergence4["Mz"]["Mean"]["real"])
#     M2_perp4 = np.array(convergence4["M2_perp"]["Mean"]["real"])

#     axs[0].grid(alpha=0.3)
#     axs[0].plot(iters1, energy1, label="Gen 1")
#     axs[0].plot(iters2, energy2, label="Gen 2")
#     axs[0].plot(iters3, energy3, label="Gen 3")
#     axs[0].plot(iters4, energy4, label="Gen 4")
#     axs[0].legend(fontsize=12)

#     axs[3].grid(alpha=0.3)
#     axs[3].plot(iters3, mx_mean3, label=f"$M_x$")
#     axs[3].plot(iters3, my_mean3, label=f"$M_y$")
#     axs[3].plot(iters3, mz_mean3, label=f"$M_z$")
#     axs[3].plot(iters3, M2_perp3, label=f"$M^2_\\perp$")
#     axs[3].legend(fontsize=12)


#     #-------------------------------------
#     # Correlator decay plots
#     #-------------------------------------

#     # Extract correlators from observables for all generations
#     d1_list = []
#     c1_list = []
#     c1_err_list = []
#     for key in observables1.keys():
#         if key.startswith("Cxy_graph_r"):
#             d = int(key.split("Cxy_graph_r")[1])
#             d1_list.append(d)
#             c1_list.append(observables1[key]["mean"])
#             c1_err_list.append(np.sqrt(observables1[key]["variance"]))

#     d1_corner_list = []
#     c1_corner_list = []
#     c1_corner_err_list = []
#     for key in observables1.keys():
#         if key.startswith("Cxy_corner_d"):
#             d = int(key.split("Cxy_corner_d")[1])
#             d1_corner_list.append(d)
#             c1_corner_list.append(observables1[key]["mean"])
#             c1_corner_err_list.append(np.sqrt(observables1[key]["variance"]))
    
#     d2_list = []
#     c2_list = []
#     c2_err_list = []
#     for key in observables2.keys():
#         if key.startswith("Cxy_graph_r"):
#             d = int(key.split("Cxy_graph_r")[1])
#             d2_list.append(d)
#             c2_list.append(observables2[key]["mean"])
#             c2_err_list.append(np.sqrt(observables2[key]["variance"]))
    
#     d2_corner_list = []
#     c2_corner_list = []
#     c2_corner_err_list = []
#     for key in observables2.keys():
#         if key.startswith("Cxy_corner_d"):
#             d = int(key.split("Cxy_corner_d")[1])
#             d2_corner_list.append(d)
#             c2_corner_list.append(observables2[key]["mean"])
#             c2_corner_err_list.append(np.sqrt(observables2[key]["variance"]))
    
#     d3_list = []
#     c3_list = []
#     c3_err_list = []
#     for key in observables3.keys():
#         if key.startswith("Cxy_graph_r"):
#             d = int(key.split("Cxy_graph_r")[1])
#             d3_list.append(d)
#             c3_list.append(observables3[key]["mean"])
#             c3_err_list.append(np.sqrt(observables3[key]["variance"]))
    
#     d3_corner_list = []
#     c3_corner_list = []
#     c3_corner_err_list = []
#     for key in observables3.keys():
#         if key.startswith("Cxy_corner_d"):
#             d = int(key.split("Cxy_corner_d")[1])
#             d3_corner_list.append(d)
#             c3_corner_list.append(observables3[key]["mean"])
#             c3_corner_err_list.append(np.sqrt(observables3[key]["variance"]))
    
#     d4_list = []
#     c4_list = []
#     c4_err_list = []
#     for key in observables4.keys():
#         if key.startswith("Cxy_graph_r"):
#             d = int(key.split("Cxy_graph_r")[1])
#             d4_list.append(d)
#             c4_list.append(observables4[key]["mean"])
#             c4_err_list.append(np.sqrt(observables4[key]["variance"]))
    
#     d4_corner_list = []
#     c4_corner_list = []
#     c4_corner_err_list = []
#     for key in observables4.keys():
#         if key.startswith("Cxy_corner_d"):
#             d = int(key.split("Cxy_corner_d")[1])
#             d4_corner_list.append(d)
#             c4_corner_list.append(observables4[key]["mean"])
#             c4_corner_err_list.append(np.sqrt(observables4[key]["variance"]))
    
#     markers = ['o', 's', '^', 'd']
#     colors = ['C0', 'C1', 'C2', 'C3']
#     gen_data = [#(d1_list, c1_list, c1_err_list, 1),
#                 (d2_list, c2_list, c2_err_list, 2),
#                 (d3_list, c3_list, c3_err_list, 3),
#                 # (d4_list, c4_list, c4_err_list, 4)
#                 ]
    
#     gen_corner_data = [#(d1_corner_list, c1_corner_list, c1_corner_err_list, 1),
#                        (d2_corner_list, c2_corner_list, c2_corner_err_list, 2),
#                        (d3_corner_list, c3_corner_list, c3_corner_err_list, 3),
#                     #    (d4_corner_list, c4_corner_list, c4_corner_err_list, 4)
#                        ]
    
#     for (d_list, c_list, c_err_list, gen), marker, color in zip(gen_data, markers, colors):
#         if len(d_list) > 0:
#             d_arr = np.array(d_list)
#             c_arr = np.array(c_list)
#             c_err_arr = np.array(c_err_list)
#             order = np.argsort(d_arr)
#             d_arr = d_arr[order]
#             c_arr = c_arr[order]
#             c_err_arr = c_err_arr[order]

#             axs[2].errorbar(d_arr, c_arr, yerr=c_err_arr, fmt=marker, 
#                           markersize=8, linewidth=2, markeredgewidth=2, 
#                           linestyle='-', capsize=4, color=color, 
#                           label=f'Gen {gen}')
            

#             mask = c_arr > 0
#             d_pos = d_arr[mask]
#             c_pos = c_arr[mask]
#             c_err_pos = c_err_arr[mask]
            
#             if len(d_pos) >= 2:
#                 # Power-law fit: C(r) = A * r^(-eta)
#                 log_r = np.log(d_pos)
#                 log_c = np.log(c_pos)
#                 poly = np.polyfit(log_r, log_c, 1)
#                 eta = -poly[0]
#                 A = np.exp(poly[1])
#                 print(f"Gen {gen} (graph): η = {eta:.4f}, A = {A:.4f}")

#                 # Exponential fit: C(r) = B * exp(-|r|/xi)
#                 def exp_decay(r, B, xi):
#                     return B * np.exp(-r / xi)
#                 popt, pcov = curve_fit(exp_decay, d_pos, c_pos, p0=(1.0, 1.0))
#                 B_fit, xi_fit = popt
#                 print(f"Gen {gen} (graph): B = {B_fit:.4f}, xi = {xi_fit:.4f}")


                
#                 axs[5].scatter(d_pos, c_pos, marker=marker, s=64, 
#                              color=color, label=f'Gen {gen}', zorder=3)
                
#                 r_fine = np.logspace(np.log10(d_pos.min()), np.log10(d_pos.max()), 100)
#                 r_fine_lin = np.linspace(d_pos.min(), d_pos.max(), 100)
#                 axs[5].loglog(r_fine, A * r_fine**(-eta), '--', 
#                             linewidth=2.5, color=color, 
#                             label=rf'$\eta_{gen} = {eta:.3f}$', zorder=2)
#                 # axs[5].loglog(r_fine_lin, exp_decay(r_fine_lin, B_fit, xi_fit), '-', 
#                             # linewidth=2, color=color, alpha=0.5,
#                             # label=rf'$\xi_{gen} = {xi_fit:.3f}$', zorder=1)
    
#     axs[2].legend(fontsize=12, framealpha=0.9)
#     axs[2].grid(alpha=0.3)
#     axs[5].legend(fontsize=12, framealpha=0.9)
#     axs[5].grid(alpha=0.3, which='both', linestyle='--')
    
#     # Plot corner correlators with different linestyle
#     for (d_list, c_list, c_err_list, gen), marker, color in zip(gen_corner_data, markers, colors):
#         if len(d_list) > 0:
#             d_arr = np.array(d_list)
#             c_arr = np.array(c_list)
#             c_err_arr = np.array(c_err_list)
#             order = np.argsort(d_arr)
#             d_arr = d_arr[order]
#             c_arr = c_arr[order]
#             c_err_arr = c_err_arr[order]
            
#             axs[2].errorbar(d_arr, c_arr, yerr=c_err_arr, fmt=marker, 
#                           markersize=8, linewidth=2, markeredgewidth=2, 
#                           linestyle='--', capsize=4, color=color, 
#                           label=f'Gen {gen} (corner)', alpha=0.7)
            

#             mask = c_arr > 0
#             d_pos = d_arr[mask]
#             c_pos = c_arr[mask]
#             c_err_pos = c_err_arr[mask]
            
#             if len(d_pos) >= 2:
#                 # Power-law fit: C(r) = A * r^(-eta)
#                 log_r = np.log(d_pos)
#                 log_c = np.log(c_pos)
#                 poly = np.polyfit(log_r, log_c, 1)
#                 eta = -poly[0]
#                 A = np.exp(poly[1])
#                 print(f"Gen {gen} (corner): η = {eta:.4f}, A = {A:.4f}")

#                 # Exponential fit: C(r) = B * exp(-|r|/xi)
#                 def exp_decay(r, B, xi):
#                     return B * np.exp(-r / xi)
#                 popt, pcov = curve_fit(exp_decay, d_pos, c_pos, p0=(1.0, 1.0))
#                 B_fit, xi_fit = popt
#                 print(f"Gen {gen} (corner): B = {B_fit:.4f}, xi = {xi_fit:.4f}")
                
#                 # Plot data points with hollow markers
#                 axs[5].scatter(d_pos, c_pos, marker=marker, s=64, 
#                              facecolors='none', edgecolors=color, linewidths=2,
#                              label=f'Gen {gen} (corner)', zorder=3, alpha=0.7)
                
#                 # Plot fit line
#                 r_fine = np.logspace(np.log10(d_pos.min()), np.log10(d_pos.max()), 100)
#                 r_fine_lin = np.linspace(d_pos.min(), d_pos.max(), 100)
#                 axs[5].loglog(r_fine, A * r_fine**(-eta), ':', 
#                             linewidth=2.5, color=color, 
#                             label=rf'$\eta_{{{gen}c}} = {eta:.3f}$', zorder=2, alpha=0.7)
#                 # axs[5].loglog(r_fine_lin, exp_decay(r_fine_lin, B_fit, xi_fit), '-', 
#                             # linewidth=1.5, color=color, alpha=0.4,
#                             # label=rf'$\xi_{{{gen}c}} = {xi_fit:.3f}$', zorder=1)


#     plt.tight_layout()
#     plt.savefig("xy_results.png", dpi=300)
#     plt.show()

#     # Create separate linear-linear plot for exponential fits
#     fig2, ax = plt.subplots(1, 1, figsize=(10, 7))
#     ax.set_xlabel(r'$r$', fontsize=18)
#     ax.set_ylabel(r'$C_{xy}(r)$', fontsize=18)
#     ax.grid(alpha=0.3)

#     # Plot graph correlators
#     for (d_list, c_list, c_err_list, gen), marker, color in zip(gen_data, markers, colors):
#         if len(d_list) > 0:
#             d_arr = np.array(d_list)
#             c_arr = np.array(c_list)
#             c_err_arr = np.array(c_err_list)
#             order = np.argsort(d_arr)
#             d_arr = d_arr[order]
#             c_arr = c_arr[order]
#             c_err_arr = c_err_arr[order]

#             mask = c_arr > 0
#             d_pos = d_arr[mask]
#             c_pos = c_arr[mask]
#             c_err_pos = c_err_arr[mask]
            
#             if len(d_pos) >= 2:
#                 # Exponential fit: C(r) = B * exp(-|r|/xi)
#                 def exp_decay(r, B, xi):
#                     return B * np.exp(-r / xi)
#                 popt, pcov = curve_fit(exp_decay, d_pos, c_pos, p0=(1.0, 1.0))
#                 B_fit, xi_fit = popt
                
#                 ax.scatter(d_pos, c_pos, marker=marker, s=80, 
#                           color=color, zorder=3)
                
#                 r_fine = np.linspace(d_pos.min(), d_pos.max(), 100)
#                 ax.plot(r_fine, exp_decay(r_fine, B_fit, xi_fit), '-', 
#                        linewidth=2.5, color=color,
#                        label=rf'Gen {gen}: $\xi = {xi_fit:.3f}$', zorder=2)

#     # Plot corner correlators
#     for (d_list, c_list, c_err_list, gen), marker, color in zip(gen_corner_data, markers, colors):
#         if len(d_list) > 0:
#             d_arr = np.array(d_list)
#             c_arr = np.array(c_list)
#             c_err_arr = np.array(c_err_list)
#             order = np.argsort(d_arr)
#             d_arr = d_arr[order]
#             c_arr = c_arr[order]
#             c_err_arr = c_err_arr[order]

#             mask = c_arr > 0
#             d_pos = d_arr[mask]
#             c_pos = c_arr[mask]
#             c_err_pos = c_err_arr[mask]
            
#             if len(d_pos) >= 2:
#                 # Exponential fit: C(r) = B * exp(-|r|/xi)
#                 def exp_decay(r, B, xi):
#                     return B * np.exp(-r / xi)
#                 popt, pcov = curve_fit(exp_decay, d_pos, c_pos, p0=(1.0, 1.0))
#                 B_fit, xi_fit = popt
                
#                 ax.scatter(d_pos, c_pos, marker=marker, s=80,
#                           facecolors='none', edgecolors=color, linewidths=2,
#                           zorder=3, alpha=0.7)
                
#                 r_fine = np.linspace(d_pos.min(), d_pos.max(), 100)
#                 ax.plot(r_fine, exp_decay(r_fine, B_fit, xi_fit), '--', 
#                        linewidth=2.5, color=color, alpha=0.7,
#                        label=rf'Gen {gen} (corner): $\xi = {xi_fit:.3f}$', zorder=2)

#     ax.legend(fontsize=11, framealpha=0.9, loc='best')
#     plt.tight_layout()
#     plt.savefig("xy_exponential_fits.png", dpi=300)
#     plt.show()


#create_figure()