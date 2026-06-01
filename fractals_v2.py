#%%
import scipy.sparse as sp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def kinetic_square(G, pbc):
        t = 1
        Nx = 3**G
        Ny = Nx
        ex=np.ones(Nx)
        ey=np.ones(Ny)
        if pbc==False:
            A=sp.spdiags(2*[ey],[1,-1],Ny,Ny) 
            L=sp.spdiags(2*[ex],[1,-1],Nx,Nx)
        else:
            A=sp.spdiags(4*[ey],[1,-1,Ny-1,1-Ny],Ny,Ny) 
            L=sp.spdiags(4*[ex],[1,-1,Nx-1,-(Nx-1)],Nx,Nx)
        
        hop_x = sp.kron(L,sp.eye(Ny))
        hop_y = sp.block_diag(Nx*[A])
        hopping = hop_x + hop_y
        Kinetic = t*hopping 
        return Kinetic, hop_x, hop_y

def kinetic_carpet(G):
    k = np.zeros((8,8))
    for i in range(2):
        k[i,i+1] = 1
        k[i+5,i+1+5] = 1
    for i in np.arange(0,7,4):
        k[i,i+3]=1
    for i in range(2,4):
        k[i,i+2]=1
    k = sp.lil_matrix(k + k.T)
    I = sp.eye(8)
    data = [1.,1.,1.]
    rows_x = [2,4,7]
    cols_x = [0,3,5]
    rows_y = [5,6,7]
    cols_y = [0,1,2]
    X = sp.coo_matrix(([1,1,1], (rows_x, cols_x)), shape=(8, 8), dtype=float)
    Y = sp.coo_matrix((data, (rows_y, cols_y)), shape=(8, 8), dtype=float)
    A = np.array([[1,0],
                [0,0],
                [0,1]])
    scale_x = X.copy()
    scale_y = Y.copy()
    for i in range (G-1):
        if i >= 1:
            X = sp.kron(scale_x,X)
            Y = sp.kron(scale_y,Y)
        k = sp.kron(I,k)
        a = sp.kron(A,Y)
        l = sp.kron(A.T,Y)
        X_T = X.T
        B = sp.bmat([[None, X,None],
                [X_T, None, X],
                [None, X_T, None]])
        d = sp.bmat([[B, a, None],
                [a.T, None, l],
                [None, l.T, B]])  
        k = k + d
    return k

def carpet(G, n=4):
    if G == 0:
        Nx = n
        Ny = n
        N = Nx * Ny
        adj = sp.lil_matrix((N, N), dtype=int)
        
        def idx(i, j):
            return i * Ny + j
        
        for i in range(Nx):
            for j in range(Ny):
                site = idx(i, j)
                if i > 0:
                    adj[site, idx(i-1, j)] = 1
                if i < Nx - 1:
                    adj[site, idx(i+1, j)] = 1
                if j > 0:
                    adj[site, idx(i, j-1)] = 1
                if j < Ny - 1:
                    adj[site, idx(i, j+1)] = 1
        return adj.tocsr()
    
    A_base = kinetic_carpet(G)
    N_base = A_base.shape[0]
    
    def make_square_adj(n):
        ex = np.ones(n)
        line = sp.spdiags([ex, ex], [1, -1], n, n)
        return sp.kron(sp.eye(n), line) + sp.kron(line, sp.eye(n))
    
    A_internal = make_square_adj(n)
    N_block = n**2
    
    internal_bonds = sp.kron(sp.eye(N_base), A_internal)
    external_bonds = sp.kron(A_base, sp.eye(N_block))
    
    return (internal_bonds + external_bonds).tocsr()

def unite_carpet_coordinates(G):

    N = 3**G
    coords = []

    def recurse(x0, y0, size, level):
        if level == 0:
            coords.append((x0, y0))
            return
        step = size // 3
        block_order = [
            (0,0), (1,0), (2,0),
            (0,1),        (2,1),
            (0,2), (1,2), (2,2)
        ]

        for bx, by in block_order:
            nx = x0 + bx * step
            ny = y0 + by * step
            recurse(nx, ny, step, level - 1)

    recurse(0, 0, N, G)

    xs, ys = zip(*coords)
    return np.array(xs), np.array(ys)

def carpet_coordinates(G, n=4):

    base_x, base_y = unite_carpet_coordinates(G)
    num_base_sites = len(base_x)

    offsets = np.arange(n)

    dx, dy = np.meshgrid(offsets, offsets)
    dx = dx.flatten()
    dy = dy.flatten()
    num_offsets = len(dx)

    final_x = np.repeat(base_x * n, num_offsets) + np.tile(dx, num_base_sites)
    final_y = np.repeat(base_y * n, num_offsets) + np.tile(dy, num_base_sites)
    
    return final_x, final_y

def triangular_patch(n):
    if n < 1:
        raise ValueError("n must be >= 3. Use n=3 for the minimal 3-site triangle.")

    sqrt3 = np.sqrt(3)

    index = {}
    coords = []

    k = 0
    for r in range(n):
        for c in range(n - r):
            index[(r, c)] = k
            x = c + 0.5 * r
            y = (sqrt3 / 2) * r
            coords.append((x, y))
            k += 1

    N = len(coords)
    A = sp.lil_matrix((N, N), dtype=int)

    # Add NN bonds of triangular lattice
    for (r, c), i in index.items():

        # right neighbor
        nb = (r, c + 1)
        if nb in index:
            j = index[nb]
            A[i, j] = 1
            A[j, i] = 1

        # upper-left neighbor
        nb = (r + 1, c)
        if nb in index:
            j = index[nb]
            A[i, j] = 1
            A[j, i] = 1

        # upper-right neighbor
        nb = (r + 1, c - 1)
        if nb in index:
            j = index[nb]
            A[i, j] = 1
            A[j, i] = 1

    coords = np.array(coords, dtype=float)
    x = coords[:, 0]
    y = coords[:, 1]

    return A.tolil(), x, y

def triangular_gasket(G,n):
    k = triangular_patch(n)[0]

    k = k.tocsr()
    right_vertex = n -1

    for i in range(G):
        Nsites_i = (3**i*(n*(n+1)-3)+3)//2
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
        k[Nsites_i, right_vertex] = 1
        k[right_vertex, Nsites_i + n-1] = 1
        k[Nsites_i + n-1, right_vertex] = 1
        k[right_vertex, Nsites_i + n-1] = 1
        k[Nsites_i + n-1, right_vertex] = 1
        # connecting k1 and k3
        k[Nsites_i - 1, 2*(Nsites_i-1)] = 1
        k[Nsites_i - 1, 2*(Nsites_i-1) + n-1] = 1
        k[2*(Nsites_i-1), Nsites_i - 1] = 1
        k[2*(Nsites_i-1) + n-1, Nsites_i - 1] = 1
        # connecting k2 and k3
        k[2*(Nsites_i-1) + right_vertex-1, 2*(Nsites_i-1)-1] = 1
        k[2*(Nsites_i-1)-1, 2*(Nsites_i-1) + right_vertex-1] = 1
        if n == 2 and i >= 1:
            k[2*(Nsites_i-1) + right_vertex-1, 2*(Nsites_i-1) - 3] = 1
            k[2*(Nsites_i-1) - 3, 2*(Nsites_i-1) + right_vertex - 1] = 1
        else:
            k[2*(Nsites_i-1) + right_vertex-1, 2*(Nsites_i-1) - 2] = 1
            k[2*(Nsites_i-1) - 2, 2*(Nsites_i-1) + right_vertex-1] = 1
            k[2*(Nsites_i-1) + right_vertex-1, 2*(Nsites_i-1) - 2] = 1
            k[2*(Nsites_i-1) - 2, 2*(Nsites_i-1) + right_vertex - 1] = 1

        if i == 0:
            patch1 = ns_i
            patch2 = keep_2 + Nsites_i - 1
            patch2 = np.concatenate((patch2, [right_vertex, 2*(Nsites_i-1) + right_vertex-1]))
            patch3 = keep_3 + 2 * Nsites_i - 3 
            patch3 = np.concatenate((patch3, [Nsites_i -1]))
        else:
            patch1 = all_patches
            patch2 = patch1 + Nsites_i-1
            patch2[0][0] = right_vertex
            patch2[2][-2] = 2*(Nsites_i-1) + right_vertex-1
            patch3 = patch1 + 2 * Nsites_i - 3
            patch3[0][0] = Nsites_i - 1
        all_patches = np.vstack((patch1, patch2, patch3))
        right_vertex += Nsites_i - 1
    if G == 0:
        all_patches = np.array([[0,1,2],[3,2,4],[4,5,2]])
    return k, all_patches

def honeycomb_gasket_coordinates(G, n):
    _, xa, ya =  triangular_patch(n)
    _, xb, yb =  triangular_patch(n-1)
    xb = np.asarray(xb+0.5, dtype=np.float64).copy()
    yb = np.asarray(yb+np.sqrt(3)/6, dtype=np.float64).copy()

    for i in range(G):
        side_a = xa.max()
        height_a = ya.max()

        xa_left = np.delete(xa, [0, -1]) + side_a
        ya_left = np.delete(ya, [0, -1])
        xa_top = np.delete(xa, [0]) + side_a * 0.5
        ya_top = np.delete(ya, [0]) + height_a
        
        xb_left = xb + side_a
        yb_left = yb
        xb_top = xb + side_a * 0.5
        yb_top = yb + height_a
        xa = np.concatenate((xa, xa_left, xa_top))
        ya = np.concatenate((ya, ya_left, ya_top))
        xb = np.concatenate((xb, xb_left, xb_top))
        yb = np.concatenate((yb, yb_left, yb_top))
    return xa, ya, xb, yb

def honeycomb_patch(n):
    Nsites_a = n*(n-1)//2
    Nsites_b = n*(n+1)//2
    k = sp.lil_matrix((Nsites_a, Nsites_b), dtype=int)
    nb_i = 0
    na_i = 0
    for i in range(n-1):
        for j in range(n-1-i):    
            k[j+nb_i, j+na_i] = 1
            k[j+nb_i, j+na_i + 1] = 1
            k[j+nb_i, j+na_i + n - i] = 1
        nb_i+=n-1-i 
        na_i+=n-i

    return k.tocsr()

def honeycomb_gasket(G,n):
    k = honeycomb_patch(n)
    right_vertex = n - 1
    Nsites_a = (n+1)*(n)//2
    Nsites_b = n*(n+1)//2
    # patch1_a = np.array([0,1,2])
    # patch2_a = np.array([3,2,4])
    # patch3_a = np.array([4,5,2])

    # patches_a = np.array([[0,1,2],[3,2,4],[4,5,2]])
    # all_patches_a = patches_a
    for i in range(G):
        Nsites_i_a = (3**i*(n*(n+1)-3)+3)//2
        Nsites_ip = (3**(i+2)+3)//2 - 1
        Nsites_ih = 3**i*(n-1)*n//2
        ns_i_a = np.arange(Nsites_i_a)
        delete_2 = [0, -1]
        delete_3 = [0]
        keep_2 = np.delete(ns_i_a, delete_2)
        keep_3 = np.delete(ns_i_a, delete_3)
        k1 = k.copy()
        k2 = k[:,:][:,1:-1]
        k3 = k[:,:][:,1:]
        k = sp.block_diag((k1, k2, k3), format='csr')
        
        # connecting k1 and k2
        k[Nsites_ih, right_vertex] = 1
        # connecting k1 and k3
        k[-Nsites_ih, Nsites_i_a-1] = 1
        # connecting k2 and k3
        k[2*Nsites_ih-1, 2*(Nsites_i_a-1) + right_vertex-1] = 1
        
        if i == 0:
            patch1_a = ns_i_a
            patch2_a = keep_2 + Nsites_i_a - 1
            patch2_a = np.concatenate((patch2_a, [right_vertex, 2*(Nsites_i_a-1) + right_vertex-1]))
            patch3_a = keep_3 + 2 * Nsites_i_a - 3 
            patch3_a = np.concatenate((patch3_a, [Nsites_i_a -1]))
        else:
            patch1_a = all_patches_a
            patch2_a = patch1_a + Nsites_i_a-1
            patch2_a[0][0] = right_vertex
            patch2_a[2][-2] = 2*(Nsites_i_a-1) + right_vertex-1
            patch3_a = patch1_a + 2 * Nsites_i_a - 3
            patch3_a[0][0] = Nsites_i_a - 1
        all_patches_a = np.vstack((patch1_a, patch2_a, patch3_a))

        right_vertex += Nsites_i_a - 1

    all_patches_b = (np.arange(3**(G)*n*(n-1)//2)  + all_patches_a.max() + 1 ).reshape((-1,n*(n-1)//2))
    all_patches = np.hstack((all_patches_a, all_patches_b))

    k = sp.bmat([[None,k.T],    
                [k,None]])
    return k.tocsr(), all_patches

def plot_graph(A,x,y):
    graph = nx.from_numpy_array(A)  # Replace with your actual adjacency matrix
    pos = dict(enumerate(zip(x, y)))
    nx.draw(
        graph,
        pos=pos,
        node_size=1,
        edge_color='black',
        width=0.8,
        connectionstyle=f'arc3,rad={0.2}'
    )
    plt.axis('equal')

# %%
# G = 3
# n = 1
# x,y = carpet_coordinates(G, n)
# adj = carpet(G, n)
# plot_graph(adj, x, y)
#  # %%
# G = 4
# n = 2
# adj, all_patches = triangular_gasket(G,n)
# xa, ya, _, _ = honeycomb_gasket_coordinates(G,n)
# plt.plot(xa, ya)
# plot_graph(adj, xa, ya)
# %%
# G = 3
# n = 2
# xa, ya, xb, yb = honeycomb_gasket_coordinates(G,n)
# x,y = np.hstack((xa,xb)), np.hstack((ya,yb))
# adj, all_patches = honeycomb_gasket(G,n)
# plot_graph(adj, x, y)
# plt.scatter(x,y,c='k')
# colors = plt.cm.tab10(np.arange(len(all_patches)))
# for i in range(1):
#     for j in all_patches[i]:
#         plt.scatter(x[j],y[j],c=colors[i])
#%%
# G = nx.from_numpy_array(adj, create_using=nx.MultiDiGraph)
# pos = {i: (xa[i], ya[i]) for i in range(len(xa))}
# nx.draw_networkx(
#     G,
#     pos=pos,
#     node_size=10,
#     with_labels=False,
#     arrows=True,
#     arrowsize=1/2,
#     connectionstyle="arc3,rad=0.15",
#     arrowstyle=patches.ArrowStyle(
#         "Fancy",
#         head_length=2.5/2,
#         head_width=1./2,
#         tail_width=.2/2,
#     ),
# )

# %%
# G = 3
# n = 3 
# k = triangular_patch(n)[0]

# k = k.tocsr()
# right_vertex = n -1
# for i in range(G):
#     print(right_vertex)
#     Nsites_i = (3**i*(n*(n+1)-3)+3)//2
#     ns_i = np.arange(Nsites_i)
#     delete_2 = [0, -1]
#     delete_3 = [0]
#     keep_2 = np.delete(ns_i, delete_2)
#     keep_3 = np.delete(ns_i, delete_3)
#     k1 = k.copy() 
#     k2 = k[keep_2,:][:,keep_2]
#     k3 = k[keep_3,:][:,keep_3]
#     k = sp.block_diag((k1, k2, k3), format='csr').tolil()
#     # connecting k1 and k2
#     k[right_vertex, Nsites_i] = 1
#     k[Nsites_i, right_vertex] = 1
#     k[right_vertex, Nsites_i + n-1] = 1
#     k[Nsites_i + n-1, right_vertex] = 1
#     k[right_vertex, Nsites_i + n-1] = 1
#     k[Nsites_i + n-1, right_vertex] = 1
#     # connecting k1 and k3
#     k[Nsites_i - 1, 2*(Nsites_i-1)] = 1
#     k[Nsites_i - 1, 2*(Nsites_i-1) + n-1] = 1
#     k[2*(Nsites_i-1), Nsites_i - 1] = 1
#     k[2*(Nsites_i-1) + n-1, Nsites_i - 1] = 1
#     # connecting k2 and k3
#     k[2*(Nsites_i-1) + right_vertex-1, 2*(Nsites_i-1)-1] = 1
#     k[2*(Nsites_i-1) + right_vertex-1, 2*(Nsites_i-1) - 2] = 1
#     k[2*(Nsites_i-1)-1, 2*(Nsites_i-1) + right_vertex-1] = 1
#     k[2*(Nsites_i-1) - 2, 2*(Nsites_i-1) + right_vertex-1] = 1
#     if i == 0:
#         patch1 = ns_i
#         patch2 = keep_2 + Nsites_i - 1
#         patch2 = np.concatenate((patch2, [right_vertex, 2*(Nsites_i-1) + right_vertex-1]))
#         patch3 = keep_3 + 2 * Nsites_i - 3 
#         patch3 = np.concatenate((patch3, [Nsites_i -1]))
#     else:
#         patch1 = all_patches
#         patch2 = patch1 + Nsites_i-1
#         patch2[0][0] = right_vertex
#         patch2[2][-2] = 2*(Nsites_i-1) + right_vertex-1
#         patch3 = patch1 + 2 * Nsites_i - 3
#         patch3[0][0] = Nsites_i - 1
#     all_patches = np.vstack((patch1, patch2, patch3))
#     right_vertex += Nsites_i - 1
    
# # %%

# %%
