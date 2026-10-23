import json
from collections import defaultdict

import flax
import networkx as nx
import numpy as np
import netket as nk
from netket.operator.spin import sigmax, sigmay, sigmaz


def save_vstate(vstate, filename):
    with open(filename, "wb") as file:
        file.write(flax.serialization.to_bytes(vstate))


def save_metadata(filename, model_config, sampler_config, hamiltonian_config, optimizer_config=None, final_results=None, **extra_sections):
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
    return {"Mx": Mx, "My": My, "Mz": Mz, "M2_perp": Mx @ Mx + My @ My}


def to_networkx(g):
    G = nx.Graph()
    G.add_nodes_from(range(g.n_nodes))
    G.add_edges_from(g.edges())
    return G


def get_distance_shells(g):
    """Pairs i < j grouped by graph distance."""
    G = to_networkx(g)
    shells = defaultdict(list)
    for i in range(g.n_nodes):
        for j, d in nx.single_source_shortest_path_length(G, i).items():
            if j > i and d > 0:
                shells[d].append((i, j))
    return shells


def get_euclidean_distance_shells_all(positions):
    """Pairs i < j grouped by Euclidean distance (rounded to 4 decimals)."""
    N = len(positions)
    shells = defaultdict(list)
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.round(np.linalg.norm(positions[i] - positions[j]), decimals=4)
            shells[dist].append((i, j))
    return shells


def get_extremal_corners(positions):
    """Bottom-left, bottom-right and top corner of the gasket."""
    x, y = positions[:, 0], positions[:, 1]
    bottom = np.where(np.isclose(y, y.min()))[0]
    bottom_left = bottom[np.argmin(x[bottom])]
    bottom_right = bottom[np.argmax(x[bottom])]
    top = np.argmax(y)
    return int(bottom_left), int(bottom_right), int(top)


def build_corner_path_correlators(hi, g, positions):
    """C_xy along the shortest paths between the three outer corners, averaged per
    graph distance. These paths run along the outer boundary of the gasket."""
    G = to_networkx(g)
    bottom_left, bottom_right, top = get_extremal_corners(positions)
    corners = [(bottom_left, bottom_right), (bottom_left, top), (bottom_right, top)]

    correlators = defaultdict(list)
    for source, target in corners:
        path = nx.shortest_path(G, source=source, target=target)
        for a in range(len(path)):
            for b in range(a + 1, len(path)):
                i, j = path[a], path[b]
                d = nx.shortest_path_length(G, i, j)
                correlators[d].append(sigmax(hi, i) @ sigmax(hi, j) + sigmay(hi, i) @ sigmay(hi, j))

    return {f"Cxy_corner_d{d}": sum(ops) / len(ops) for d, ops in sorted(correlators.items())}
