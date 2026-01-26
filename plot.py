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




# Load graph structures

edges1 = np.genfromtxt("edges1.txt", dtype=int)
edges_list1 = [tuple(map(int, e)) for e in edges1]
vertices1 = np.genfromtxt("vertices1.txt")
N1 = len(vertices1)
convergence1 = json.load(open("sierpinski_gen1.log"))
observables1 = json.load(open("sierpinski_gen1_observables.json"))


edges2 = np.genfromtxt("edges2.txt", dtype=int)
edges_list2 = [tuple(map(int, e)) for e in edges2]
vertices2 = np.genfromtxt("vertices2.txt")
N2 = len(vertices2)
convergence2 = json.load(open("sierpinski_gen2.log"))
observables2 = json.load(open("sierpinski_gen2_observables.json"))


edges3 = np.genfromtxt("edges3.txt", dtype=int)
edges_list3 = [tuple(map(int, e)) for e in edges3]
vertices3 = np.genfromtxt("vertices3.txt")
N3 = len(vertices3)
convergence3 = json.load(open("sierpinski_gen3.log"))
observables3 = json.load(open("sierpinski_gen3_observables.json"))


edges4 = np.genfromtxt("edges4.txt", dtype=int)
edges_list4 = [tuple(map(int, e)) for e in edges4]
vertices4 = np.genfromtxt("vertices4.txt")
N4 = len(vertices4)
convergence4 = json.load(open("sierpinski_gen4.log"))
observables4 = json.load(open("sierpinski_gen4_observables.json"))


def create_figure():
    # Create 2x3 subplot figure
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()
    

    xlabels = [r"$Iterations$", r"", r"$r$", r"$Iterations$", r"", r"$r$"]
    ylabels = [r"$E$", r"", r"$C_{xy}$", r"", r"", r"$C_{xy}$"]

    for i in range(6):
        axs[i].set_xlabel(xlabels[i])
        axs[i].set_ylabel(ylabels[i])

    for ax in axs:
        ax.xaxis.label.set_size(18)
        ax.yaxis.label.set_size(18)

    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[4].set_xticks([])
    axs[4].set_yticks([])
    axs[1].set_xticklabels([])
    axs[1].set_yticklabels([])
    axs[4].set_xticklabels([])
    axs[4].set_yticklabels([])

    axs[5].set_xscale('log')
    axs[5].set_yscale('log')

    #-------------------------------------
    # Site resolved magnetization^2 plots
    #-------------------------------------

    # Generate random test magnetization data
    np.random.seed(42)  # For reproducibility

    mx3 = np.array([observables3[f"Mx_{i}"]["mean"] for i in range(N3)])
    mx4 = np.array([observables4[f"Mx_{i}"]["mean"] for i in range(N4)])
    m2_perp_3 = np.array([observables3[f"M2_perp_{i}"]["mean"] for i in range(N3)])
    m2_perp_4 = np.array([observables4[f"M2_perp_{i}"]["mean"] for i in range(N4)])

    e3_variance = observables3["E"]["variance"]
    e4_variance = observables4["E"]["variance"]


    x3, y3 = vertices3[:, 0], vertices3[:, 1]
    for i, j in edges_list3:
        axs[1].plot([x3[i], x3[j]], [y3[i], y3[j]], 'k-', alpha=0.3, linewidth=0.8)
    scatter3 = axs[1].scatter(x3, y3, c=m2_perp_3, s=30, zorder=5, 
                              edgecolors='black', linewidth=0.5,
                              cmap='viridis')
    axs[1].set_aspect('equal')
    axs[1].set_title(f"Gen 3: E variance = {e3_variance:.2e}", fontsize=14)
    axs[1].grid(False)
    
    cbar2 = fig.colorbar(scatter3, ax=axs[1], orientation='horizontal', 
                       location='bottom', pad=0.05, fraction=0.046, aspect=20)
    # cbar2.set_label(r'$\langle \sigma^x_i \rangle$', fontsize=14)
    cbar2.set_label(r'$\langle M^2_{\perp,i} \rangle$', fontsize=14)

    x4, y4 = vertices4[:, 0], vertices4[:, 1]
    for i, j in edges_list4:
        axs[4].plot([x4[i], x4[j]], [y4[i], y4[j]], 'k-', alpha=0.3, linewidth=0.8)
    scatter4 = axs[4].scatter(x4, y4, c=m2_perp_4, s=30, zorder=5, 
                              edgecolors='black', linewidth=0.5,
                              cmap='viridis')
    axs[4].set_aspect('equal')
    axs[4].set_title(f"Gen 4: E variance = {e4_variance:.2e}", fontsize=14)
    axs[4].grid(False)
    
    cbar = fig.colorbar(scatter4, ax=axs[4], orientation='horizontal', 
                       location='bottom', pad=0.05, fraction=0.046, aspect=20)
    # cbar.set_label(r'$\langle \sigma^x_i \rangle$', fontsize=14)
    cbar.set_label(r'$\langle M^2_{\perp,i} \rangle$', fontsize=14)

    #-------------------------------------
    # Energy and Mag. convergence plots
    #-------------------------------------

    iters1 = convergence1["Energy"]["iters"]
    iters2 = convergence2["Energy"]["iters"]
    iters3 = convergence3["Energy"]["iters"]
    iters4 = convergence4["Energy"]["iters"]

    energy1 = np.array(convergence1["Energy"]["Mean"]["real"])#/N1
    mx_mean1 = np.array(convergence1["Mx"]["Mean"]["real"])
    my_mean1 = np.array(convergence1["My"]["Mean"]["real"])
    mz_mean1 = np.array(convergence1["Mz"]["Mean"]["real"])
    M2_perp1 = np.array(convergence1["M2_perp"]["Mean"]["real"])

    energy2 = np.array(convergence2["Energy"]["Mean"]["real"])#/N2
    mx_mean2 = np.array(convergence2["Mx"]["Mean"]["real"])
    my_mean2 = np.array(convergence2["My"]["Mean"]["real"])
    mz_mean2 = np.array(convergence2["Mz"]["Mean"]["real"])
    M2_perp2 = np.array(convergence2["M2_perp"]["Mean"]["real"])

    energy3 = np.array(convergence3["Energy"]["Mean"]["real"])#/N3
    mx_mean3 = np.array(convergence3["Mx"]["Mean"]["real"])
    my_mean3 = np.array(convergence3["My"]["Mean"]["real"])
    mz_mean3 = np.array(convergence3["Mz"]["Mean"]["real"])
    M2_perp3 = np.array(convergence3["M2_perp"]["Mean"]["real"])

    energy4 = np.array(convergence4["Energy"]["Mean"]["real"])#/N4
    mx_mean4 = np.array(convergence4["Mx"]["Mean"]["real"])
    my_mean4 = np.array(convergence4["My"]["Mean"]["real"])
    mz_mean4 = np.array(convergence4["Mz"]["Mean"]["real"])
    M2_perp4 = np.array(convergence4["M2_perp"]["Mean"]["real"])

    axs[0].grid(alpha=0.3)
    axs[0].plot(iters1, energy1, label="Gen 1")
    axs[0].plot(iters2, energy2, label="Gen 2")
    axs[0].plot(iters3, energy3, label="Gen 3")
    axs[0].plot(iters4, energy4, label="Gen 4")
    axs[0].legend(fontsize=12)

    axs[3].grid(alpha=0.3)
    axs[3].plot(iters3, mx_mean3, label=f"$M_x$")
    axs[3].plot(iters3, my_mean3, label=f"$M_y$")
    axs[3].plot(iters3, mz_mean3, label=f"$M_z$")
    axs[3].plot(iters3, M2_perp3, label=f"$M^2_\\perp$")
    axs[3].legend(fontsize=12)


    #-------------------------------------
    # Correlator decay plots
    #-------------------------------------

    # Extract correlators from observables for all generations
    d1_list = []
    c1_list = []
    c1_err_list = []
    for key in observables1.keys():
        if key.startswith("Cxy_graph_r"):
            d = int(key.split("Cxy_graph_r")[1])
            d1_list.append(d)
            c1_list.append(observables1[key]["mean"])
            c1_err_list.append(np.sqrt(observables1[key]["variance"]))

    d1_corner_list = []
    c1_corner_list = []
    c1_corner_err_list = []
    for key in observables1.keys():
        if key.startswith("Cxy_corner_d"):
            d = int(key.split("Cxy_corner_d")[1])
            d1_corner_list.append(d)
            c1_corner_list.append(observables1[key]["mean"])
            c1_corner_err_list.append(np.sqrt(observables1[key]["variance"]))
    
    d2_list = []
    c2_list = []
    c2_err_list = []
    for key in observables2.keys():
        if key.startswith("Cxy_graph_r"):
            d = int(key.split("Cxy_graph_r")[1])
            d2_list.append(d)
            c2_list.append(observables2[key]["mean"])
            c2_err_list.append(np.sqrt(observables2[key]["variance"]))
    
    d2_corner_list = []
    c2_corner_list = []
    c2_corner_err_list = []
    for key in observables2.keys():
        if key.startswith("Cxy_corner_d"):
            d = int(key.split("Cxy_corner_d")[1])
            d2_corner_list.append(d)
            c2_corner_list.append(observables2[key]["mean"])
            c2_corner_err_list.append(np.sqrt(observables2[key]["variance"]))
    
    d3_list = []
    c3_list = []
    c3_err_list = []
    for key in observables3.keys():
        if key.startswith("Cxy_graph_r"):
            d = int(key.split("Cxy_graph_r")[1])
            d3_list.append(d)
            c3_list.append(observables3[key]["mean"])
            c3_err_list.append(np.sqrt(observables3[key]["variance"]))
    
    d3_corner_list = []
    c3_corner_list = []
    c3_corner_err_list = []
    for key in observables3.keys():
        if key.startswith("Cxy_corner_d"):
            d = int(key.split("Cxy_corner_d")[1])
            d3_corner_list.append(d)
            c3_corner_list.append(observables3[key]["mean"])
            c3_corner_err_list.append(np.sqrt(observables3[key]["variance"]))
    
    d4_list = []
    c4_list = []
    c4_err_list = []
    for key in observables4.keys():
        if key.startswith("Cxy_graph_r"):
            d = int(key.split("Cxy_graph_r")[1])
            d4_list.append(d)
            c4_list.append(observables4[key]["mean"])
            c4_err_list.append(np.sqrt(observables4[key]["variance"]))
    
    d4_corner_list = []
    c4_corner_list = []
    c4_corner_err_list = []
    for key in observables4.keys():
        if key.startswith("Cxy_corner_d"):
            d = int(key.split("Cxy_corner_d")[1])
            d4_corner_list.append(d)
            c4_corner_list.append(observables4[key]["mean"])
            c4_corner_err_list.append(np.sqrt(observables4[key]["variance"]))
    
    markers = ['o', 's', '^', 'd']
    colors = ['C0', 'C1', 'C2', 'C3']
    gen_data = [#(d1_list, c1_list, c1_err_list, 1),
                (d2_list, c2_list, c2_err_list, 2),
                (d3_list, c3_list, c3_err_list, 3),
                # (d4_list, c4_list, c4_err_list, 4)
                ]
    
    gen_corner_data = [#(d1_corner_list, c1_corner_list, c1_corner_err_list, 1),
                       (d2_corner_list, c2_corner_list, c2_corner_err_list, 2),
                       (d3_corner_list, c3_corner_list, c3_corner_err_list, 3),
                    #    (d4_corner_list, c4_corner_list, c4_corner_err_list, 4)
                       ]
    
    for (d_list, c_list, c_err_list, gen), marker, color in zip(gen_data, markers, colors):
        if len(d_list) > 0:
            d_arr = np.array(d_list)
            c_arr = np.array(c_list)
            c_err_arr = np.array(c_err_list)
            order = np.argsort(d_arr)
            d_arr = d_arr[order]
            c_arr = c_arr[order]
            c_err_arr = c_err_arr[order]

            axs[2].errorbar(d_arr, c_arr, yerr=c_err_arr, fmt=marker, 
                          markersize=8, linewidth=2, markeredgewidth=2, 
                          linestyle='-', capsize=4, color=color, 
                          label=f'Gen {gen}')
            

            mask = c_arr > 0
            d_pos = d_arr[mask]
            c_pos = c_arr[mask]
            c_err_pos = c_err_arr[mask]
            
            if len(d_pos) >= 2:
                # Power-law fit: C(r) = A * r^(-eta)
                log_r = np.log(d_pos)
                log_c = np.log(c_pos)
                poly = np.polyfit(log_r, log_c, 1)
                eta = -poly[0]
                A = np.exp(poly[1])
                print(f"Gen {gen} (graph): η = {eta:.4f}, A = {A:.4f}")

                # Exponential fit: C(r) = B * exp(-|r|/xi)
                def exp_decay(r, B, xi):
                    return B * np.exp(-r / xi)
                popt, pcov = curve_fit(exp_decay, d_pos, c_pos, p0=(1.0, 1.0))
                B_fit, xi_fit = popt
                print(f"Gen {gen} (graph): B = {B_fit:.4f}, xi = {xi_fit:.4f}")


                
                axs[5].scatter(d_pos, c_pos, marker=marker, s=64, 
                             color=color, label=f'Gen {gen}', zorder=3)
                
                r_fine = np.logspace(np.log10(d_pos.min()), np.log10(d_pos.max()), 100)
                r_fine_lin = np.linspace(d_pos.min(), d_pos.max(), 100)
                axs[5].loglog(r_fine, A * r_fine**(-eta), '--', 
                            linewidth=2.5, color=color, 
                            label=rf'$\eta_{gen} = {eta:.3f}$', zorder=2)
                # axs[5].loglog(r_fine_lin, exp_decay(r_fine_lin, B_fit, xi_fit), '-', 
                            # linewidth=2, color=color, alpha=0.5,
                            # label=rf'$\xi_{gen} = {xi_fit:.3f}$', zorder=1)
    
    axs[2].legend(fontsize=12, framealpha=0.9)
    axs[2].grid(alpha=0.3)
    axs[5].legend(fontsize=12, framealpha=0.9)
    axs[5].grid(alpha=0.3, which='both', linestyle='--')
    
    # Plot corner correlators with different linestyle
    for (d_list, c_list, c_err_list, gen), marker, color in zip(gen_corner_data, markers, colors):
        if len(d_list) > 0:
            d_arr = np.array(d_list)
            c_arr = np.array(c_list)
            c_err_arr = np.array(c_err_list)
            order = np.argsort(d_arr)
            d_arr = d_arr[order]
            c_arr = c_arr[order]
            c_err_arr = c_err_arr[order]
            
            axs[2].errorbar(d_arr, c_arr, yerr=c_err_arr, fmt=marker, 
                          markersize=8, linewidth=2, markeredgewidth=2, 
                          linestyle='--', capsize=4, color=color, 
                          label=f'Gen {gen} (corner)', alpha=0.7)
            

            mask = c_arr > 0
            d_pos = d_arr[mask]
            c_pos = c_arr[mask]
            c_err_pos = c_err_arr[mask]
            
            if len(d_pos) >= 2:
                # Power-law fit: C(r) = A * r^(-eta)
                log_r = np.log(d_pos)
                log_c = np.log(c_pos)
                poly = np.polyfit(log_r, log_c, 1)
                eta = -poly[0]
                A = np.exp(poly[1])
                print(f"Gen {gen} (corner): η = {eta:.4f}, A = {A:.4f}")

                # Exponential fit: C(r) = B * exp(-|r|/xi)
                def exp_decay(r, B, xi):
                    return B * np.exp(-r / xi)
                popt, pcov = curve_fit(exp_decay, d_pos, c_pos, p0=(1.0, 1.0))
                B_fit, xi_fit = popt
                print(f"Gen {gen} (corner): B = {B_fit:.4f}, xi = {xi_fit:.4f}")
                
                # Plot data points with hollow markers
                axs[5].scatter(d_pos, c_pos, marker=marker, s=64, 
                             facecolors='none', edgecolors=color, linewidths=2,
                             label=f'Gen {gen} (corner)', zorder=3, alpha=0.7)
                
                # Plot fit line
                r_fine = np.logspace(np.log10(d_pos.min()), np.log10(d_pos.max()), 100)
                r_fine_lin = np.linspace(d_pos.min(), d_pos.max(), 100)
                axs[5].loglog(r_fine, A * r_fine**(-eta), ':', 
                            linewidth=2.5, color=color, 
                            label=rf'$\eta_{{{gen}c}} = {eta:.3f}$', zorder=2, alpha=0.7)
                # axs[5].loglog(r_fine_lin, exp_decay(r_fine_lin, B_fit, xi_fit), '-', 
                            # linewidth=1.5, color=color, alpha=0.4,
                            # label=rf'$\xi_{{{gen}c}} = {xi_fit:.3f}$', zorder=1)


    plt.tight_layout()
    plt.savefig("xy_results.png", dpi=300)
    plt.show()

    # Create separate linear-linear plot for exponential fits
    fig2, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.set_xlabel(r'$r$', fontsize=18)
    ax.set_ylabel(r'$C_{xy}(r)$', fontsize=18)
    ax.grid(alpha=0.3)

    # Plot graph correlators
    for (d_list, c_list, c_err_list, gen), marker, color in zip(gen_data, markers, colors):
        if len(d_list) > 0:
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
            
            if len(d_pos) >= 2:
                # Exponential fit: C(r) = B * exp(-|r|/xi)
                def exp_decay(r, B, xi):
                    return B * np.exp(-r / xi)
                popt, pcov = curve_fit(exp_decay, d_pos, c_pos, p0=(1.0, 1.0))
                B_fit, xi_fit = popt
                
                ax.scatter(d_pos, c_pos, marker=marker, s=80, 
                          color=color, zorder=3)
                
                r_fine = np.linspace(d_pos.min(), d_pos.max(), 100)
                ax.plot(r_fine, exp_decay(r_fine, B_fit, xi_fit), '-', 
                       linewidth=2.5, color=color,
                       label=rf'Gen {gen}: $\xi = {xi_fit:.3f}$', zorder=2)

    # Plot corner correlators
    for (d_list, c_list, c_err_list, gen), marker, color in zip(gen_corner_data, markers, colors):
        if len(d_list) > 0:
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
            
            if len(d_pos) >= 2:
                # Exponential fit: C(r) = B * exp(-|r|/xi)
                def exp_decay(r, B, xi):
                    return B * np.exp(-r / xi)
                popt, pcov = curve_fit(exp_decay, d_pos, c_pos, p0=(1.0, 1.0))
                B_fit, xi_fit = popt
                
                ax.scatter(d_pos, c_pos, marker=marker, s=80,
                          facecolors='none', edgecolors=color, linewidths=2,
                          zorder=3, alpha=0.7)
                
                r_fine = np.linspace(d_pos.min(), d_pos.max(), 100)
                ax.plot(r_fine, exp_decay(r_fine, B_fit, xi_fit), '--', 
                       linewidth=2.5, color=color, alpha=0.7,
                       label=rf'Gen {gen} (corner): $\xi = {xi_fit:.3f}$', zorder=2)

    ax.legend(fontsize=11, framealpha=0.9, loc='best')
    plt.tight_layout()
    plt.savefig("xy_exponential_fits.png", dpi=300)
    plt.show()




create_figure()


'''
# Euclidean distance correlators
d1_euclidean_list = []
c1_euclidean_list = []
c1_euclidean_err_list = []
for key in observables1.keys():
    if key.startswith("Cxy_euclidean_r"):
        d = float(key.split("Cxy_euclidean_r")[1])
        d1_euclidean_list.append(d)
        c1_euclidean_list.append(observables1[key]["mean"])
        c1_euclidean_err_list.append(np.sqrt(observables1[key]["variance"]))

d_arr = np.array(d1_euclidean_list)
c_arr = np.array(c1_euclidean_list)
order = np.argsort(d_arr)
d_arr = d_arr[order]
c_arr = c_arr[order]
'''
