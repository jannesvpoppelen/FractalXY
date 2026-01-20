#%%
import os
rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))

if rank == 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.environ["JAX_PLATFORM_NAME"] = "gpu" if rank == 0 else "cpu"

import netket as nk 
import scipy.sparse as sp
import numpy as np
import json
import matplotlib.pyplot as plt
import netket.nn as nknn
import flax.linen as nn
import jax.numpy as jnp
import time
from scipy.sparse.linalg import eigsh
import itertools
import sys 
import jax
import os
import logging


print(f"JAX is using: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
# os.environ["JAX_PLATFORM_NAME"] = "cpu"

#%%

# --- GASKET FUNCTIONS ---
def t_gasket(G):
    Nsites = (3**(G+1)+3)//2
    k = sp.lil_matrix((3,3),dtype=int)
    k[0,1] = 1
    k[0,2] = 1
    k[1,2] = 1
    k = k.tocsr()
    right_vertex = 1

    for i in range(G):
        Nsites_i = (3**(i+1)+3)//2
        ns_i = np.arange(Nsites_i)
        delete_2 = [0, -1]
        delete_3 = [0]
        keep_2 = np.delete(ns_i, delete_2)
        keep_3 = np.delete(ns_i, delete_3)
        k1 = k.copy() 
        k2 = k[keep_2,:][:,keep_2]
        k3 = k[keep_3,:][:,keep_3]
        k = sp.block_diag((k1, k2, k3), format='csr').tolil()
        # connecting k1 and k2
        k[right_vertex, Nsites_i] = 1
        k[right_vertex, Nsites_i + 1] = 1
        # connecting k1 and k3
        k[Nsites_i - 1, 2*(Nsites_i-1)] = 1
        k[Nsites_i - 1, 2*(Nsites_i-1) + 1] = 1
        #connecting k2 and k3
        k[2*(Nsites_i-1) + right_vertex-1, 2*(Nsites_i-1)-1] = 1
        if i>0:
            k[2*(Nsites_i-1) + right_vertex-1, 2*(Nsites_i-1) - 3] = 1
        right_vertex += Nsites_i - 1
    k = k + k.T
    return k

def h_gasket(G):
    k = sp.lil_matrix((1,3),dtype=int)
    k[0,0] = 1
    k[0,1] = 1
    k[0,2] = 1
    k = k 
    k = k.tocsr()
    right_vertex = 1
    Nsites_ih = 0
    for i in range(G):
        Nsites_i = (3**(i+1)+3)//2
        Nsites_ih = 3**i
        Nsites_ihp = 3**(i-1)
        ns_i = np.arange(Nsites_i)
        delete_2 = [0, -1]
        delete_3 = [0]
        keep_2 = np.delete(ns_i, delete_2)
        keep_3 = np.delete(ns_i, delete_3)
        k1 = k.copy() 
        k2 = k[:,:][:,keep_2]
        k3 = k[:,:][:,keep_3]
        k = sp.block_diag((k1, k2, k3), format='csr').tolil()
        # connecting k1 and k2
        k[Nsites_ih, right_vertex] = 1
        # connecting k1 and k3
        k[2*Nsites_ih, Nsites_i - 1] = 1
        #connecting k2 and k3
        k[2*Nsites_ih -1 ,2*(Nsites_i-1) + right_vertex-1] = 1
        # if i>0:
        #     k[2*(Nsites_i-1) + right_vertex-1, 2*(Nsites_i-1) - 3] = np.exp(1j * 2 * np.pi / 3)
        right_vertex += Nsites_i - 1
    k = sp.bmat([[None,k.T],
                [k,None]])
    return k.tocsr() 

def gasket_coordinates(G):
    """
    Generates the (x, y) coordinates for a Sierpinski Gasket of generation G,
    following the order defined by the kinetic matrix construction logic.
    The final coordinates form a properly scaled, equilateral triangle centered at (0, 0).
    """
    SQRT3 = np.sqrt(3)
    G = G + 1 
    # --- 1. Recursive Function to Determine Node Order (Same as before) ---
    def generate_recursive_coords(g, current_coords):
        # Base coordinates for G=1 (Standard Normalized Equilateral: 
        # V0=(0,0), V1=(1,0), V2=(0.5, SQRT3/2))
        if g == 1:
            return [current_coords[0], current_coords[1], current_coords[2]]

        sub_coords = generate_recursive_coords(g - 1, current_coords)
        Nsites_i = len(sub_coords)
        
        V0 = current_coords[0] 
        V1 = current_coords[1] 
        V2 = current_coords[2] 

        # K1 (Bottom-Left sub-gasket) - Full K_i
        K1_coords = [V0 + 0.5 * (p - V0) for p in sub_coords]

        # K2' (Bottom-Right sub-gasket) - K_i excluding index 0 (BL) and Nsites_i-1 (Top)
        K2_full = [V1 + 0.5 * (p - V1) for p in sub_coords]
        K2_kept_indices = np.arange(1, Nsites_i - 1)
        K2_coords = [K2_full[i] for i in K2_kept_indices]

        # K3'' (Top sub-gasket) - K_i excluding index 0 (BL)
        K3_full = [V2 + 0.5 * (p - V2) for p in sub_coords]
        K3_kept_indices = np.arange(1, Nsites_i) 
        K3_coords = [K3_full[i] for i in K3_kept_indices]

        # Combine in the specific order K1, K2', K3''
        return K1_coords + K2_coords + K3_coords

    # Initial corners for the G=1 base
    base_coords = {
        0: np.array([0.0, 0.0]), 
        1: np.array([1.0, 0.0]), 
        2: np.array([0.5, SQRT3 / 2])
    }

    ordered_coords_list = generate_recursive_coords(G, base_coords)
    
    X_std = np.array([p[0] for p in ordered_coords_list])
    Y_std = np.array([p[1] for p in ordered_coords_list])

    # --- 2. Scaling and Centering Transformation ---
    # A. Scale the coordinates to have a side length of 2*G (from -G to G in X)
    # The standard Gasket spans [0, 1] in X, [0, SQRT3/2] in Y.
    X_scaled = X_std * G * 2
    Y_scaled = Y_std * G * 2
    
    # B. Center the Gasket at (0, 0)
    # Translation vector to center the shape at (0, 0):
    X_center = G 
    Y_center = G * SQRT3 / 2 # Half the height
    
    X_final = X_scaled - X_center
    Y_final = Y_scaled - Y_center

    return X_final, Y_final

def gasket_center_coordinates(G):
    """
    Returns the (x, y) coordinates of the centers of the smallest triangles
    of the Sierpinski gasket of generation G.

    Fully consistent with gasket_coordinates(G).
    """

    SQRT3 = np.sqrt(3)
    Gp = G + 1  # same convention as gasket_coordinates

    centers = []

    def recurse(g, V0, V1, V2):
        if g == 1:
            centers.append((V0 + V1 + V2) / 3)
            return

        M01 = 0.5 * (V0 + V1)
        M02 = 0.5 * (V0 + V2)
        M12 = 0.5 * (V1 + V2)

        recurse(g - 1, V0,  M01, M02)
        recurse(g - 1, M01, V1,  M12)
        recurse(g - 1, M02, M12, V2)

    # base triangle (identical to gasket_coordinates)
    V0 = np.array([0.0, 0.0])
    V1 = np.array([1.0, 0.0])
    V2 = np.array([0.5, SQRT3 / 2])

    recurse(Gp, V0, V1, V2)

    centers = np.array(centers)

    # --- EXACT same scaling and centering ---
    centers *= Gp * 2
    centers[:, 0] -= Gp
    centers[:, 1] -= Gp * SQRT3 / 2

    return centers[:, 0], centers[:, 1]

def honeycomb_gasket(G):
    kv = t_gasket(G)
    kc = h_gasket(G)
    kh = sp.bmat([[kv,kc.T],
                   [kc,None]])
    return kh.tocsr()

def plot_graph_from_csr(x, y, A, node_size=30, lw=1):
    plt.figure(figsize=(6, 6))

    A = A.tocoo() 

    for i, j in zip(A.row, A.col):
        if i < j:  
            plt.plot([x[i], x[j]], [y[i], y[j]], 'k-', lw=lw)

    plt.scatter(x, y, s=node_size)
    plt.axis('equal')
    plt.axis('off')
    plt.show()

def plot_netket_graph(graph, x, y):
    plt.figure(figsize=(6, 6))
    for i, j in graph.edges():
        plt.plot([x[i], x[j]], [y[i], y[j]], 'k-', lw=1)

    plt.scatter(x, y, s=30)
    plt.axis('equal')
    plt.axis('off')
    plt.show()
# %%
G = 2
xv,yv = gasket_coordinates(G)
xc,yc = gasket_center_coordinates(G)
x, y = np.hstack((xv,xc)), np.hstack((yv,yc))
Nsites = len(x)
A = h_gasket(G)
A = A.tocoo()   
edges = list(zip(A.row, A.col))
edges = [(i, j) for i, j in edges if i < j]

hi = nk.hilbert.Spin(s=1/2, N=Nsites) 
graph = nk.graph.Graph(edges=edges, n_nodes=Nsites)
plot_netket_graph(graph, x, y)

# HAMILTONIAN 
H = nk.operator.LocalOperator(hi, dtype=complex)
Jij = 1
for i, j in edges:
    H += -Jij * nk.operator.spin.sigmax(hi, i) @ nk.operator.spin.sigmax(hi, j)
    H += -Jij * nk.operator.spin.sigmay(hi, i) @ nk.operator.spin.sigmay(hi, j)
    # H += -0.25 * Jij * nk.operator.spin.sigmaz(hi, i) @ nk.operator.spin.sigmaz(hi, j)

Mx = sum(nk.operator.spin.sigmax(hi, i) for i in range(Nsites)) / Nsites
My = sum(nk.operator.spin.sigmay(hi, i) for i in range(Nsites)) / Nsites   
Mz = sum(nk.operator.spin.sigmaz(hi, i) for i in range(Nsites)) / Nsites

# pairs = list(itertools.product(range(Nsites), range(Nsites)))
# Sxx = np.array([nk.operator.spin.sigmax(hi, i)@nk.operator.spin.sigmax(hi, j) for i,j in pairs])
# Syy = nk.operator.spin.sigmay(hi, i)@nk.operator.spin.sigmay(hi, j)
# Sxy = nk.operator.spin.sigmax(hi, i)@nk.operator.spin.sigmay(hi, j)

# e_gs, psi_gs = eigsh(H.to_sparse(), k=1, which='SA')
# e_gs = e_gs[0]
# e_gs = nk.exact.lanczos_ed(H, compute_eigenvectors=False) 
# psi_gs = psi_gs.reshape(-1)
# print(f"Exact ground state energy: {e_gs:.6f}")

model = nk.models.RBM(alpha=1)
sampler = nk.sampler.MetropolisLocal(hi, n_chains=64)
vstate= nk.vqs.MCState(sampler, model, n_samples=2**12, n_discard_per_chain=1000)
optimizer = nk.optimizer.Sgd(learning_rate=0.01)
gs = nk.driver.VMC_SR(hamiltonian=H, optimizer=optimizer, variational_state=vstate, diag_shift=0.0001)
# gs = nk.driver.VMC(H, optimizer, variational_state=vstate, preconditioner=nk.optimizer.SR(diag_shift=0.01))

name = f'hg_G={G}'
start = time.time()
gs.run(out=name, obs={"Mx": Mx, "My": My, "Mz": Mz}, n_iter=250)
end = time.time()

print("### RBM calculation")
print("Has", vstate.n_parameters, "parameters")
print("The RBM calculation took", end - start, "seconds")
# %%
name = 'hg_G=4'
data = json.load(open(f"{name}.log"))
iters = data["Energy"]["iters"]
energy = data["Energy"]["Mean"]["real"]
mx_s = data["Mx"]["Mean"]
my_s = data["My"]["Mean"]["real"]
mz_s = data["Mz"]["Mean"]

#%%
plt.plot(iters, energy, label="VMC Energy")
plt.hlines(e_gs, xmin=0, xmax=max(iters), colors='r', linestyles='dashed', label="Exact Ground State Energy")
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.legend()
plt.show()
