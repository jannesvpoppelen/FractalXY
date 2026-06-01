"""XY model simulation on fractal and regular lattices using NetKet."""
#%%
import numpy as np
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax
import jax.random as jr
import netket as nk
from netket.operator.spin import sigmax, sigmay, sigmaz
import flax
from flax import linen as nn
import optax
import time

from utils import *
from fractals import *

import matplotlib.pyplot as plt

import netket as nk

import jax.numpy as jnp

print(jax.devices())


from einops import rearrange

print(f"NetKet {nk.__version__}, Jax {jax.__version__}")
print(f"JAX is using: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")


def save_vstate(vstate, filename):
    with open(filename, 'wb') as file:
        file.write(flax.serialization.to_bytes(vstate))

def build_hamiltonian(hi, g, J=1.0, Jz=0.0, hx=0.0, hy = 0.0):
    H = nk.operator.LocalOperator(hi, dtype=complex)
    for i, j in g.edges():
        H += -J * (sigmax(hi, i) @ sigmax(hi, j) + sigmay(hi, i) @ sigmay(hi, j))
        if Jz != 0.0:
            H += -Jz * (sigmaz(hi, i) @ sigmaz(hi, j))
    if hx != 0.0 or hy != 0.0:
        for i in range(g.n_nodes):
            H += -hx * sigmax(hi, i) -hy * sigmay(hi, i)
    return H

# G = 0
# n = 3 
# A, patches = triangular_gasket(G, n=3)
# A = A.tocoo()   
# edges = list(zip(A.row, A.col))
# edges_list = [(i, j) for i, j in edges if i < j]

# g_sierpinski = nk.graph.Graph(edges_list)

# # d_model = 32  # embedding dimension
# d_model = 16
# # initialize M batch of spin configurations, for a fractal geometry with N nodes
# M = 200
# N = g_sierpinski.n_nodes
# seed = 0
# key = jax.random.key(seed)
# key, subkey = jax.random.split(key)
# spin_configs = jax.random.randint(subkey, shape=(M, N), minval=0, maxval=2) * 2 - 1
# print(f"{spin_configs.shape = }")
# n_patches = patches.shape[0]


class Embed(nn.Module):
    d_model: int  # dimensionality of the embedding space
    patches: np.ndarray  # indices of patches to extract
    param_dtype = jnp.float64

    def setup(self):
        self.embed = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=self.param_dtype,
        )

    def __call__(self, x):
        x = x[:, self.patches]
        x = self.embed(x)

        return x
    
# embed_module = Embed(d_model, patches)
# key, subkey = jax.random.split(key)
# params_embed = embed_module.init(subkey, spin_configs)

# # apply the embedding module to the spin configurations
# embedded_configs = embed_module.apply(params_embed, spin_configs)

# print(f"{embedded_configs.shape = }")
#Multi-Head Attention 
class FactoredAttention(nn.Module):
    n_patches: int  # lenght of the input sequence
    d_model: int  # dimensionality of the embedding space (d in the equations)

    def setup(self):
        self.alpha = self.param(
            "alpha", nn.initializers.xavier_uniform(), (self.n_patches, self.n_patches)
        )
        self.V = self.param(
            "V", nn.initializers.xavier_uniform(), (self.d_model, self.d_model)
        )

    def __call__(self, x):
        y = jnp.einsum("i j, a b, M j b-> M i a", self.alpha, self.V, x)
        return y

from functools import partial

@partial(jax.vmap, in_axes=(None, 0, None), out_axes=1)
@partial(jax.vmap, in_axes=(None, None, 0), out_axes=1)
def roll2d(spins, i, j):
    side = int(spins.shape[-1] ** 0.5)
    spins = spins.reshape(spins.shape[0], side, side)
    spins = jnp.roll(jnp.roll(spins, i, axis=-2), j, axis=-1)
    return spins.reshape(spins.shape[0], -1)


class FMHA(nn.Module):
    d_model: int  # dimensionality of the embedding space
    n_heads: int  # number of heads
    n_patches: int  # lenght of the input sequence
    transl_invariant: bool = False
    param_dtype = jnp.float64

    def setup(self):
        self.v = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=self.param_dtype,
        )
        self.W = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=self.param_dtype,
        )
        if self.transl_invariant:
            self.alpha = self.param(
                "alpha",
                nn.initializers.xavier_uniform(),
                (self.n_heads, self.n_patches),
                self.param_dtype,
            )
            sq_n_patches = int(self.n_patches**0.5)
            assert sq_n_patches * sq_n_patches == self.n_patches
            self.alpha = roll2d(
                self.alpha, jnp.arange(sq_n_patches), jnp.arange(sq_n_patches)
            )
            self.alpha = self.alpha.reshape(self.n_heads, -1, self.n_patches)
        else:
            self.alpha = self.param(
                "alpha",
                nn.initializers.xavier_uniform(),
                (self.n_heads, self.n_patches, self.n_patches),
                self.param_dtype,
            )

    def __call__(self, x):
        # apply the value matrix in paralell for each head
        v = self.v(x)

        # split the representations of the different heads
        v = rearrange(
            v,
            "batch n_patches (n_heads d_eff) -> batch n_patches n_heads d_eff",
            n_heads=self.n_heads,
        )

        # factored attention mechanism
        v = rearrange(
            v, "batch n_patches n_heads d_eff -> batch n_heads n_patches d_eff"
        )
        x = jnp.matmul(self.alpha, v)
        x = rearrange(
            x, "batch n_heads n_patches d_eff  -> batch n_patches n_heads d_eff"
        )

        # concatenate the different heads
        x = rearrange(
            x, "batch n_patches n_heads d_eff ->  batch n_patches (n_heads d_eff)"
        )

        # the representations of the different heads are combined together
        x = self.W(x)

        return x

# # test Factored MultiHead Attention module
# n_heads = 8  # number of heads
# n_patches = embedded_configs.shape[1]  # lenght of the input sequence

# # initialize the Factored Multi-Head Attention module
# fmha_module = FMHA(d_model, n_heads, n_patches)

# key, subkey = jax.random.split(key)
# params_fmha = fmha_module.init(subkey, embedded_configs)

# # apply the Factored Multi-Head Attention module to the embedding vectors
# attention_vectors = fmha_module.apply(params_fmha, embedded_configs)

# print(f"{attention_vectors.shape = }")

class EncoderBlock(nn.Module):
    d_model: int  # dimensionality of the embedding space
    n_heads: int  # number of heads
    n_patches: int  # lenght of the input sequence
    transl_invariant: bool = False
    param_dtype = jnp.float64

    def setup(self):
        self.attn = FMHA(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_patches=self.n_patches,
            transl_invariant=self.transl_invariant,
        )

        self.layer_norm_1 = nn.LayerNorm(param_dtype=self.param_dtype)
        self.layer_norm_2 = nn.LayerNorm(param_dtype=self.param_dtype)

        self.ff = nn.Sequential(
            [
                nn.Dense(
                    4 * self.d_model,
                    kernel_init=nn.initializers.xavier_uniform(),
                    param_dtype=self.param_dtype,
                ),
                nn.gelu,
                nn.Dense(
                    self.d_model,
                    kernel_init=nn.initializers.xavier_uniform(),
                    param_dtype=self.param_dtype,
                ),
            ]
        )

    def __call__(self, x):
        x = x + self.attn(self.layer_norm_1(x))

        x = x + self.ff(self.layer_norm_2(x))
        return x

class Encoder(nn.Module):
    num_layers: int  # number of layers
    d_model: int  # dimensionality of the embedding space
    n_heads: int  # number of heads
    n_patches: int  # lenght of the input sequence
    transl_invariant: bool = False

    def setup(self):
        self.layers = [
            EncoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_patches=self.n_patches,
                transl_invariant=self.transl_invariant,
            )
            for _ in range(self.num_layers)
        ]

    def __call__(self, x):

        for l in self.layers:
            x = l(x)

        return x

# # test Transformer Encoder module
# num_layers = 4  # number of layers

# # initialize the Factored Multi-Head Attention module
# encoder_module = Encoder(num_layers, d_model, n_heads, n_patches)

# key, subkey = jax.random.split(key)
# params_encoder = encoder_module.init(subkey, embedded_configs)

# # apply the Factored Multi-Head Attention module to the embedding vectors
# x = embedded_configs
# y = encoder_module.apply(params_encoder, x)

# print(f"{y.shape = }")

log_cosh = (
    nk.nn.activation.log_cosh
)  # Logarithm of the hyperbolic cosine, implemented in a more stable way



class OuputHead(nn.Module):
    d_model: int  # dimensionality of the embedding space
    param_dtype = jnp.float64

    def setup(self):
        self.out_layer_norm = nn.LayerNorm(param_dtype=self.param_dtype)

        self.norm2 = nn.LayerNorm(
            use_scale=True, use_bias=True, param_dtype=self.param_dtype
        )
        self.norm3 = nn.LayerNorm(
            use_scale=True, use_bias=True, param_dtype=self.param_dtype
        )

        self.output_layer0 = nn.Dense(
            self.d_model,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )
        self.output_layer1 = nn.Dense(
            self.d_model,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )

    def __call__(self, x):

        z = self.out_layer_norm(x.sum(axis=1))

        out_real = self.norm2(self.output_layer0(z))
        out_imag = self.norm3(self.output_layer1(z))

        out = out_real + 1j * out_imag

        return jnp.sum(log_cosh(out), axis=-1)

class ViT(nn.Module):
    num_layers: int  # number of layers
    d_model: int  # dimensionality of the embedding space
    n_heads: int  # number of heads
    patches: np.ndarray  # indices of patches to extract
    transl_invariant: bool = False

    @nn.compact
    def __call__(self, spins):
        x = jnp.atleast_2d(spins)

        Ns = x.shape[-1]  # number of sites
        n_patches = len(self.patches)  # lenght of the input sequence

        x = Embed(d_model=self.d_model, patches = self.patches)(x)

        y = Encoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_patches=n_patches,
            transl_invariant=self.transl_invariant,
        )(x)

        log_psi = OuputHead(d_model=self.d_model)(y)

        return log_psi


def train_vit_single(
    hilbert,
    hamiltonian,
    patches,
    num_layers,
    d_model,
    n_heads,
    N_samples=2048,
    N_opt=800,
    early_stop_patience=150,
    seed=0,
    verbose=True,
):
    """Train ViT with early stopping based on energy plateau.

    Args:
        hilbert: NetKet Hilbert space
        hamiltonian: NetKet Hamiltonian
        patches: patch indices for embedding
        num_layers, d_model, n_heads: architecture parameters
        N_samples: number of MC samples
        N_opt: max iterations
        early_stop_patience: stop if energy doesn't improve for N iterations
        seed: random seed
        verbose: print progress

    Returns:
        dict with keys: vstate, log, final_energy, final_variance, n_params,
                       stopped_at_iter, convergence_data
    """
    import time
    start_time = time.time()

    key = jax.random.key(seed)
    patches_tuple = tuple(map(tuple, patches))

    # Initialize ViT module
    vit_module = ViT(
        num_layers=num_layers,
        d_model=d_model,
        n_heads=n_heads,
        patches=patches_tuple,
        transl_invariant=False,
    )

    # Generate initial spin configs for initialization
    N = hilbert.size
    M = 100
    key, subkey = jax.random.split(key)
    spin_configs = jax.random.randint(subkey, shape=(M, N), minval=0, maxval=2) * 2 - 1

    key, subkey = jax.random.split(key)
    params = vit_module.init(subkey, spin_configs)

    # Sampler
    sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=512)

    # Optimizer with schedule
    lr_schedule = optax.exponential_decay(
        init_value=1e-2,
        transition_steps=100,
        decay_rate=0.995,
    )
    optimizer = nk.optimizer.Sgd(learning_rate=lr_schedule)

    # Variational state
    key, subkey = jax.random.split(key, 2)
    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=vit_module,
        sampler_seed=subkey,
        n_samples=N_samples,
        n_discard_per_chain=0,
        variables=params,
        chunk_size=128,
    )

    N_params = nk.jax.tree_size(vstate.parameters)

    # VMC driver
    vmc = nk.driver.VMC_SR(
        hamiltonian=hamiltonian,
        optimizer=optimizer,
        diag_shift=1e-4,
        variational_state=vstate,
        momentum=0.8,
        mode="complex",
    )

    # Training loop with early stopping
    energies = []
    variances = []
    best_energy = float('inf')
    iterations_without_improvement = 0
    stopped_at_iter = N_opt

    for iteration in range(N_opt):
        vmc.advance(1)

        # Get current energy and variance
        E = vmc.state.expect(hamiltonian)
        energy_val = E.mean.real
        variance_val = E.variance.real

        energies.append(energy_val)
        variances.append(variance_val)

        # Early stopping check
        if energy_val < best_energy:
            best_energy = energy_val
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1

        if iterations_without_improvement >= early_stop_patience and iteration > 200:
            stopped_at_iter = iteration + 1
            if verbose:
                print(f"Early stopping at iteration {stopped_at_iter}")
            break

        if verbose and (iteration + 1) % 50 == 0:
            print(f"Iter {iteration + 1}: E={energy_val:.6f}, Var={variance_val:.6f}")

    elapsed_time = time.time() - start_time

    # Compute final estimates (average over last 50 iterations)
    final_energy = np.mean(energies[-50:])
    final_variance = np.mean(variances[-50:])

    if verbose:
        print(f"Final energy: {final_energy:.6f}, Variance: {final_variance:.6f}")
        print(f"Training time: {elapsed_time:.2f}s, Parameters: {N_params}")

    return {
        "vstate": vstate,
        "energies": np.array(energies),
        "variances": np.array(variances),
        "final_energy": final_energy,
        "final_variance": final_variance,
        "n_params": N_params,
        "stopped_at_iter": stopped_at_iter,
        "training_time": elapsed_time,
        "hamiltonian": hamiltonian,
        "hilbert": hilbert,
    }
        #%%
# # test ViT module
# # initialize the ViT module
# # vit_module = ViT(num_layers, d_model, n_heads, patches)

# # key, subkey = jax.random.split(key)
# # params = vit_module.init(subkey, spin_configs)

# # # apply the ViT module
# # log_psi = vit_module.apply(params, spin_configs)

# # print(f"{log_psi.shape = }")

# G = 0
# n = 3 
# A, patches = triangular_gasket(G, n=3)
# A = A.tocoo()   
# edges = list(zip(A.row, A.col))
# edges_list = [(i, j) for i, j in edges if i < j]

# g_sierpinski = nk.graph.Graph(edges_list)

# # initialize M batch of spin configurations, for a fractal geometry with N nodes
# M = 100
# N = g_sierpinski.n_nodes
# seed = 0
# key = jax.random.key(seed)
# key, subkey = jax.random.split(key)
# spin_configs = jax.random.randint(subkey, shape=(M, N), minval=0, maxval=2) * 2 - 1
# print(f"{spin_configs.shape = }")
# n_patches = patches.shape[0]

# seed = 0
# key = jax.random.key(seed)

# # Hilbert space of spins on the graph
# hilbert = nk.hilbert.Spin(s=1 / 2, N=g_sierpinski.n_nodes)
# hamiltonian = build_hamiltonian(hilbert, g_sierpinski, J=1, Jz=0, hx=0, hy=0)

# # Intiialize the ViT variational wave function
# patches_tuple = tuple(map(tuple, patches))
# num_layers = 2
# d_model = 30
# n_heads = 5

# vit_module = ViT(
#     num_layers=num_layers, d_model=d_model, n_heads=n_heads, patches=patches_tuple, transl_invariant=False
# )
# params = vit_module.init(subkey, spin_configs)

# # Metropolis Local Sampling
# N_samples = 1024
# n_chains = 1024
# sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_chains)

# # optimizer = nk.optimizer.Sgd(learning_rate=0.0075)


# lr_schedule = optax.exponential_decay(
#     init_value=1e-2,
#     # transition_steps=10,
#     transition_steps=100,
#     decay_rate=0.99,
# )

# optimizer = nk.optimizer.Sgd(learning_rate=lr_schedule)

# key, subkey = jax.random.split(key, 2)
# vstate = nk.vqs.MCState(
#     sampler=sampler,
#     model=vit_module,
#     sampler_seed=subkey,
#     n_samples=N_samples,
#     n_discard_per_chain=0,
#     variables=params,
#     # chunk_size=512,
#     chunk_size=128,

# )

# N_params = nk.jax.tree_size(vstate.parameters)
# print("Number of parameters = ", N_params, flush=True)

# # Variational monte carlo driver

# vmc = nk.driver.VMC_SR(
#     hamiltonian=hamiltonian,
#     optimizer=optimizer,
#     diag_shift=1e-4,
#     variational_state=vstate,
#     momentum = 0.8,  
#     mode="complex",
# )

# if N < 16:
#     E, _  = nk.exact.lanczos_ed(hamiltonian, compute_eigenvectors=True, k=3)
#     E_0 = E[0]
#     E_1 = E[1]
#     E_2 = E[2]
#     gap1 = E_1 - E_0
#     gap2 = E_2 - E_0
#     print(f"ED gaps: E1-E0={gap1}, E2-E0={gap2}")
#     print(f"ED ground state: {E_0}")
# # %%
# name = f"6_vit_{G}_{n}.mpack"
# # Optimization
# # log = nk.logging.RuntimeLog()
# # Creates '.json' (for logs) and '.mpack' 
# log = nk.logging.JsonLog(name, save_params_every=50)

# N_opt = 800
# vmc.run(n_iter=N_opt, out=log)
# #%%
# energy_per_site = log.data["Energy"]["Mean"].real / g_sierpinski.n_nodes

# print("Last value: ", energy_per_site[-1])
# metadata = {
#     "model": {
#         "num_layers": num_layers,
#         "d_model": d_model,
#         "n_heads": n_heads,
#         "patches": patches.tolist(),  # Convert numpy array to list for JSON
#         "transl_invariant": False,
#     },
#     "sampler": {
#         "n_chains": N_samples,
#         "n_samples": vstate.n_samples,
#         "n_discard_per_chain": vstate.n_discard_per_chain,
#         "seed": None,
#     },
#     "training": {
#         "hamiltonian": {"J": 1.0, "Jz": 0.0, "hx": 0.0, "hy": 0.0},
#         "generations": G,
#         "patch_size": n,
#         "n_opt": N_opt,
#         "final_energy": float(energy_per_site[-1]),
#     }
# }

# import json
# metadata_file = name.replace('.mpack', '_metadata.json')
# with open(metadata_file, 'w') as f:
#     json.dump(metadata, f, indent=2)
# print(f"✓ Saved metadata to {metadata_file}")
# # %%


# # plt.plot(energy_per_site)

# # plt.xlabel("Iterations")
# # plt.ylabel("Energy per site")

# # plt.show()
# # # save_vstate(vstate, f"vit_{G}_{n}.mpack")
# # vstate_file = '/vol/tcm13/robert_canellas_nunez/NN/FractalXY/4_vit_0_3.mpack'
# # # vstate_file = '/vol/tcm13/robert_canellas_nunez/NN/FractalXY/4_vit_0_3.mpack'
# # with open(vstate_file, 'rb') as vstatefile:
# #     vstate = flax.serialization.from_bytes(vstate, vstatefile.read())

# # %%
