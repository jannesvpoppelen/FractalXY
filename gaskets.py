#%%
#ADJACENCY MATRICES
import scipy.sparse as sp
import numpy as np

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

#COORDINATES    
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
# %%
