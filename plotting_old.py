"""
Plotting utilities for quantum XY model simulations.

This module provides visualization functions for energy convergence,
magnetizations, correlators, structure factors, and graph structures.
"""

import json
import re
import numpy as np
import matplotlib.pyplot as plt
import netket as nk


def _sanitize_filename(title):
    """Convert title to clean filename identifier."""
    # Remove special characters, keep alphanumeric and spaces
    clean = re.sub(r'[^a-zA-Z0-9\s]', '', title)
    # Replace spaces with underscores and lowercase
    return '_'.join(clean.lower().split())


def plot_training_observables(data, title="Training", E_gs=None, gen=None):
    """Plot energy, magnetizations, and in-plane magnetization during training."""
    iters = data["Energy"]["iters"]
    energy = np.array(data["Energy"]["Mean"]["real"])
    mx = np.array(data["Mx"]["Mean"]["real"])
    my = np.array(data["My"]["Mean"]["real"])
    mz = np.array(data["Mz"]["Mean"]["real"])
    m2_perp = np.array(data["M2_perp"]["Mean"]["real"])
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    # Energy
    axes[0].plot(iters, energy, linewidth=2, color='steelblue')
    if E_gs is not None:
        axes[0].axhline(E_gs, color="black", linestyle="--", label="ED", linewidth=2)
        axes[0].legend()
    axes[0].set_ylabel("Energy", fontsize=12)
    axes[0].set_title(f"{title} - Energy", fontsize=13)
    axes[0].grid(alpha=0.3)
    
    # Magnetizations
    axes[1].plot(iters, mx, label="$M_x$", alpha=0.8, color="orange", linewidth=2)
    axes[1].plot(iters, my, label="$M_y$", alpha=0.8, color="green", linewidth=2)
    axes[1].plot(iters, mz, label="$M_z$", alpha=0.8, color="red", linewidth=2)
    axes[1].set_ylabel("Magnetization", fontsize=12)
    axes[1].set_title(f"{title} - Magnetizations", fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)
    
    # In-plane magnetization
    axes[2].plot(iters, m2_perp, linewidth=2, color='purple')
    axes[2].set_ylabel(r"$M_\perp^2 = M_x^2 + M_y^2$", fontsize=12)
    axes[2].set_xlabel("Iteration", fontsize=12)
    axes[2].set_title(f"{title} - In-plane Magnetization", fontsize=13)
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    fname = f"training_gen_{gen}.png" if gen is not None else f"training_{_sanitize_filename(title)}.png"
    plt.savefig(fname, dpi=300)
    plt.show()
    
    print(f"{title} - Final Energy: {energy[-1]:.6f}")
    print(f"{title} - Final Mx: {mx[-1]:.6f}, My: {my[-1]:.6f}, Mz: {mz[-1]:.6f}")
    print(f"{title} - Final M⊥²: {m2_perp[-1]:.6f}")
    

def plot_structure_factors(data, title, gen=None):
    """Plot structure factors."""
    k_labels = []
    sk_vals = []
    for key in data.keys():
        if key.startswith("Sxy_kx"):
            parts = key.split("_")
            kx = float(parts[1][2:])
            ky = float(parts[2][2:])
            k_labels.append(f"({kx:.2f},{ky:.2f})")
            sk_vals.append(data[key]["Mean"]["real"][-1])
    
    if len(k_labels) == 0:
        print(f"No structure factors found for {title}")
        return
    
    sk_vals = np.array(sk_vals)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(np.arange(len(k_labels)), sk_vals)
    ax.set_xticks(np.arange(len(k_labels)))
    ax.set_xticklabels(k_labels, rotation=45, ha="right")
    ax.set_ylabel(r"$S_{xy}(k)$", fontsize=12)
    ax.set_title(f"{title} - Structure Factors", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    fname = f"structure_gen_{gen}.png" if gen is not None else f"structure_{_sanitize_filename(title)}.png"
    plt.savefig(fname, dpi=300)
    plt.show()


def plot_graph(g, vertices, title="Graph", z=None, gen=None):
    """Plot the graph structure with nodes and edges.
    
    Args:
        g: NetKet graph object with edges and n_nodes
        vertices: Array of shape (n_nodes, 2) with (x, y) coordinates
        title: Plot title
        z: Optional array of values to color nodes (colormap applied), must be same size as g.n_nodes
    """
    # Validate z dimensions
    if z is not None and len(z) != g.n_nodes:
        raise ValueError(f"z array length ({len(z)}) must match number of nodes ({g.n_nodes})")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    x = vertices[:, 0]
    y = vertices[:, 1]
    
    # Plot edges
    for i, j in g.edges():
        ax.plot([x[i], x[j]], [y[i], y[j]], 'k-', alpha=0.4, linewidth=1)
    
    # Plot nodes with optional coloring
    if z is not None:
        scatter = ax.scatter(x, y, c=z, s=100, zorder=5, 
                           edgecolors='black', linewidth=0.5, cmap='viridis')
        plt.colorbar(scatter, ax=ax)
    else:
        ax.scatter(x, y, c='blue', s=100, zorder=5, 
                  edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    fname = f"graph_gen_{gen}.png" if gen is not None else f"graph_{_sanitize_filename(title)}.png"
    plt.savefig(fname, dpi=300)
    plt.show()


def plot_correlator_decay(data, title, fit_range=None, gen=None):
    """
    Plot correlator decay C_xy(r) with power-law fit on log-log scale.
    
    Args:
        data: simulation data dictionary
        title: plot title
        fit_range: tuple (r_min, r_max) for fitting range, or None for all data
    """
    # Extract Cxy_graph_r{d} correlators
    d_list = []
    c_list = []
    for key in data.keys():
        if key.startswith("Cxy_graph_r"):
            d = int(key.split("Cxy_graph_r")[1])
            d_list.append(d)
            c_list.append(data[key]["Mean"]["real"][-1])
    
    if len(d_list) == 0:
        print(f"No Cxy_r observables found for {title}!")
        return
    
    # Sort by distance
    d_arr = np.array(d_list)
    c_arr = np.array(c_list)
    order = np.argsort(d_arr)
    d_arr = d_arr[order]
    c_arr = c_arr[order]
    
    # Filter positive correlators for log plots
    mask = c_arr > 0
    d_pos = d_arr[mask]
    c_pos = c_arr[mask]
    
    if len(d_pos) < 2:
        print(f"Not enough positive correlators for fitting in {title}!")
        return
    
    # Apply fit range if specified
    if fit_range is not None:
        r_min, r_max = fit_range
        fit_mask = (d_pos >= r_min) & (d_pos <= r_max)
        d_fit = d_pos[fit_mask]
        c_fit = c_pos[fit_mask]
    else:
        d_fit = d_pos
        c_fit = c_pos
    
    if len(d_fit) < 2:
        print(f"Not enough points in fit range for {title}!")
        return
    
    # Power-law fit: log(C) = log(A) - eta * log(r)
    log_r = np.log(d_fit)
    log_c = np.log(c_fit)
    poly = np.polyfit(log_r, log_c, 1)
    eta = -poly[0]
    A = np.exp(poly[1])
    
    # Create log-log plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.loglog(d_pos, c_pos, 'o', markersize=10, label='Data', color='steelblue')
    
    # Plot fit line
    r_fine = np.logspace(np.log10(d_pos.min()), np.log10(d_pos.max()), 100)
    ax.loglog(r_fine, A * r_fine**(-eta), '--', linewidth=2.5, color='red',
              label=f'Fit: $r^{{-\\eta}}$, $\\eta = {eta:.3f}$')
    
    if fit_range is not None:
        ax.axvspan(r_min, r_max, alpha=0.15, color='gray', label='Fit range')
    
    ax.set_xlabel("Distance $r$", fontsize=13)
    ax.set_ylabel(r"$C_{xy}(r) = \langle \sigma^x_i\sigma^x_j + \sigma^y_i\sigma^y_j\rangle$", fontsize=13)
    ax.set_title(f"{title} - Correlator Decay (Log-Log)", fontsize=14)
    ax.grid(alpha=0.3, which='both', linestyle='--')
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    fname = f"decay_gen_{gen}.png" if gen is not None else f"decay_{_sanitize_filename(title)}.png"
    plt.savefig(fname, dpi=300)
    plt.show()
    
    print(f"{title}: η = {eta:.4f}")


# Load graph structures
edges = np.genfromtxt("edges3.txt", dtype=int)
edges_list = [tuple(map(int, e)) for e in edges]
vertices = np.genfromtxt("vertices3.txt")
g_sierpinski = nk.graph.Graph(edges_list)

gen = 3

# Load simulation data
data_sierp = json.load(open("sierpinski2.log"))

# Plot training observables
plot_training_observables(data_sierp, "sierpinski", E_gs=None, gen=gen)

# Plot structure factors (from final observables file if available)
plot_structure_factors(data_sierp, "sierpinski", gen=gen)

print("Final Sierpinski Energy:", data_sierp["Energy"]["Mean"]["real"][-1]/g_sierpinski.n_nodes)

# Plot site-resolved magnetizations (if available in log file)
if f"Mx_0" in data_sierp:
    n_nodes = g_sierpinski.n_nodes
    mx_site = np.array([data_sierp[f"Mx_{i}"]["Mean"]["real"][-1] for i in range(n_nodes)])
    my_site = np.array([data_sierp[f"My_{i}"]["Mean"]["real"][-1] for i in range(n_nodes)])
    plot_graph(g_sierpinski, vertices, "Sierpinski - $\\langle\\sigma^x_i\\rangle$", z=mx_site, gen=gen)
    # plot_graph(g_sierpinski, vertices, "Sierpinski - $\\langle\\sigma^y_i\\rangle$", z=my_site, gen=gen)

# Plot correlator decay analysis (if correlators available)
if any(key.startswith("Cxy_graph_r") for key in data_sierp.keys()):
    plot_correlator_decay(data_sierp, "Sierpinski", gen=gen)
    # plot_correlator_decay(data_sq, "Square")