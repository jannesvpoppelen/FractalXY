"""Utility functions for graph analysis and operator construction."""

import json
import numpy as np
import networkx as nx
from collections import defaultdict
import flax
import netket as nk
from netket.operator.spin import sigmax, sigmay, sigmaz


# -----------------------------------------------------------------------
# IO utilities
# -----------------------------------------------------------------------


def save_vstate(vstate, filename):
    with open(filename, "wb") as file:
        file.write(flax.serialization.to_bytes(vstate))


def save_metadata(filename, model_config, sampler_config, hamiltonian_config, optimizer_config=None, final_results=None, **extra_sections):
    """Save metadata about model, sampler, and Hamiltonian to JSON.

    Extra keyword arguments are added as top-level sections (e.g. fractal={...}).
    """
    metadata = {
        "model": model_config,
        "sampler": sampler_config,
        "hamiltonian": hamiltonian_config,
    }
    if optimizer_config is not None:
        metadata["optimizer"] = optimizer_config
    if final_results is not None:
        metadata["final_results"] = final_results
    for key, val in extra_sections.items():
        if val is not None:
            metadata[key] = val
    with open(filename, "w") as f:
        json.dump(metadata, f, indent=2)


# -----------------------------------------------------------------------
# Physics operators
# -----------------------------------------------------------------------


def build_hamiltonian(hi, g, J=1.0, Jz=0.0, hx=0.0, hy=0.0):
    H = nk.operator.LocalOperator(hi, dtype=complex)
    for i, j in g.edges():
        H += -J * (sigmax(hi, i) @ sigmax(hi, j) + sigmay(hi, i) @ sigmay(hi, j))
        if Jz != 0.0:
            H += -Jz * (sigmaz(hi, i) @ sigmaz(hi, j))
    if hx != 0.0 or hy != 0.0:
        for i in range(g.n_nodes):
            H += -hx * sigmax(hi, i) - hy * sigmay(hi, i)
    return H


def build_magnetizations(hi, g):
    N = g.n_nodes
    Mx = sum(sigmax(hi, i) for i in range(N)) / N
    My = sum(sigmay(hi, i) for i in range(N)) / N
    Mz = sum(sigmaz(hi, i) for i in range(N)) / N

    return {
        "Mx": Mx,
        "My": My,
        "Mz": Mz,
        "M2_perp": Mx @ Mx + My @ My,
    }


# -----------------------------------------------------------------------
# Structure factor / k-space utilities
# -----------------------------------------------------------------------


def get_square_positions(L):
    coords = np.zeros((L * L, 2), dtype=float)
    idx = 0
    for y in range(L):
        for x in range(L):
            coords[idx] = (x, y)
            idx += 1
    return coords


def build_structure_factor_ops(hi, positions, k_list, prefix="Sxy"):
    """Build S_xy(k) = (1/N) Σ_ij cos(k·Δr) (σ_i^x σ_j^x + σ_i^y σ_j^y)."""
    N = len(positions)
    obs = {}

    for kx, ky in k_list:
        op = 0
        for i in range(N):
            ri_x, ri_y = positions[i]
            for j in range(N):
                rj_x, rj_y = positions[j]
                phase = np.cos(kx * (ri_x - rj_x) + ky * (ri_y - rj_y))
                op += phase * (sigmax(hi, i) @ sigmax(hi, j) +
                               sigmay(hi, i) @ sigmay(hi, j))
        obs[f"{prefix}_kx{kx:.3f}_ky{ky:.3f}"] = op / N
    return obs


def get_k_vectors(name, positions):
    if name.startswith("square"):
        return [(0.0, 0.0), (np.pi, 0.0), (0.0, np.pi), (np.pi, np.pi)]
    else:
        x_min, y_min = positions.min(axis=0)
        x_max, y_max = positions.max(axis=0)
        Lx, Ly = x_max - x_min, y_max - y_min
        return [(0.0, 0.0), (2*np.pi/(Lx+1e-9), 2*np.pi/(Ly+1e-9))]



# -----------------------------------------------------------------------
# NetworkX graph utilities
# -----------------------------------------------------------------------


def get_distance_shells(g):
    """Group unordered node pairs by shortest-path graph distance."""
    G = nx.Graph()
    G.add_nodes_from(range(g.n_nodes))
    G.add_edges_from(g.edges())
    shells = defaultdict(list)

    for i in range(g.n_nodes):
        lengths = nx.single_source_shortest_path_length(G, i)
        for j, d in lengths.items():
            if j > i and d > 0:
                shells[d].append((i, j))

    return shells

def get_euclidean_distance_shells_all(positions):
    """Group unordered node pairs by Euclidean distance."""
    N = len(positions)
    shells = defaultdict(list)

    for i in range(N):
        for j in range(i+1, N):
            dist = np.linalg.norm(positions[i] - positions[j])
            dist = np.round(dist, decimals=4)
            shells[dist].append((i, j))

    return shells

def get_shortest_path(g, source, target):
    """Get list of nodes along one shortest path from source to target."""
    G = nx.Graph()
    G.add_nodes_from(range(g.n_nodes))
    G.add_edges_from(g.edges())
    return nx.shortest_path(G, source=source, target=target)


# def get_graph_distance_shells(g, source, max_distance=None):
#     """Group nodes by graph distance from source node. """
#     G = nx.Graph()
#     G.add_nodes_from(range(g.n_nodes))
#     G.add_edges_from(g.edges())
#     shells = defaultdict(list)
#     lengths = nx.single_source_shortest_path_length(G, source)
#     
#     for j, d in lengths.items():
#         if j == source or d == 0:
#             continue
#         if max_distance is None or d <= max_distance:
#             shells[d].append((source, j))
#     
#     return shells


# def get_euclidean_distance_shells(g, positions, source, max_distance=None):
#     """Group nodes by Euclidean distance from source node."""
#     shells = defaultdict(list)
#     source_pos = positions[source]
# 
#     for j in range(g.n_nodes):
#         if j == source:
#             continue
#         dist = np.linalg.norm(positions[j] - source_pos)
#         dist = np.round(dist, decimals=4)
#         if max_distance is None or dist <= max_distance:
#             shells[dist].append((source, j))
# 
#     return shells


def build_corner_path_correlators(hi, g):
    G = nx.Graph()
    G.add_nodes_from(range(g.n_nodes))
    G.add_edges_from(g.edges())
    
    # Define corner pairs
    corners = [
        (0, 1),
        (0, 2),
        (1, 2)
    ]
    
    distance_correlators = defaultdict(list)
    
    for source, target in corners:
        path = get_shortest_path(g, source, target)
        
        # Compute correlators between all pairs along the path
        for i in range(len(path)):
            for j in range(i+1, len(path)):
                node_i = path[i]
                node_j = path[j]
                # Get actual graph distance between the pair
                d = nx.shortest_path_length(G, node_i, node_j)
                # XY correlator
                op = sigmax(hi, node_i) @ sigmax(hi, node_j) + sigmay(hi, node_i) @ sigmay(hi, node_j)
                distance_correlators[d].append(op)
    
    # Average correlators at each distance
    obs = {}
    for d in sorted(distance_correlators.keys()):
        ops = distance_correlators[d]
        avg_op = sum(ops) / len(ops)
        obs[f"Cxy_corner_d{d}"] = avg_op
    
    # print(f"Corner path correlators: distances d={list(sorted(distance_correlators.keys()))}")
    
    return obs
