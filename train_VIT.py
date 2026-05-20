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

# %%
G = 1
n = 3 
A, patches = triangular_gasket(G, n=3)
A = A.tocoo()   
edges = list(zip(A.row, A.col))
edges_list = [(i, j) for i, j in edges if i < j]

g_sierpinski = nk.graph.Graph(edges_list)

d_model = 32  # embedding dimension
# initialize M batch of spin configurations, for a fractal geometry with N nodes
M = 200
N = g_sierpinski.n_nodes
seed = 0
key = jax.random.key(seed)
key, subkey = jax.random.split(key)
spin_configs = jax.random.randint(subkey, shape=(M, N), minval=0, maxval=2) * 2 - 1
print(f"{spin_configs.shape = }")
n_patches = patches.shape[0]

# %%

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
    
embed_module = Embed(d_model, patches)
key, subkey = jax.random.split(key)
params_embed = embed_module.init(subkey, spin_configs)

# apply the embedding module to the spin configurations
embedded_configs = embed_module.apply(params_embed, spin_configs)

print(f"{embedded_configs.shape = }")
# %%
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
# %%
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
# %%
# test Factored MultiHead Attention module
n_heads = 8  # number of heads
n_patches = embedded_configs.shape[1]  # lenght of the input sequence

# initialize the Factored Multi-Head Attention module
fmha_module = FMHA(d_model, n_heads, n_patches)

key, subkey = jax.random.split(key)
params_fmha = fmha_module.init(subkey, embedded_configs)

# apply the Factored Multi-Head Attention module to the embedding vectors
attention_vectors = fmha_module.apply(params_fmha, embedded_configs)

print(f"{attention_vectors.shape = }")
# %%
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
# %%
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
# %%
# test Transformer Encoder module
num_layers = 4  # number of layers

# initialize the Factored Multi-Head Attention module
encoder_module = Encoder(num_layers, d_model, n_heads, n_patches)

key, subkey = jax.random.split(key)
params_encoder = encoder_module.init(subkey, embedded_configs)

# apply the Factored Multi-Head Attention module to the embedding vectors
x = embedded_configs
y = encoder_module.apply(params_encoder, x)

print(f"{y.shape = }")
# %%
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
# %%
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
# %%
# test ViT module
# initialize the ViT module
vit_module = ViT(num_layers, d_model, n_heads, patches)

key, subkey = jax.random.split(key)
params = vit_module.init(subkey, spin_configs)

# apply the ViT module
log_psi = vit_module.apply(params, spin_configs)

print(f"{log_psi.shape = }")
# %%
seed = 0
key = jax.random.key(seed)

# Hilbert space of spins on the graph
hilbert = nk.hilbert.Spin(s=1 / 2, N=g_sierpinski.n_nodes)
hamiltonian = build_hamiltonian(hilbert, g_sierpinski, J=1, Jz=0, hx=0, hy=0)

# Intiialize the ViT variational wave function
patches_tuple = tuple(map(tuple, patches))
vit_module = ViT(
    num_layers=4, d_model=60, n_heads=10, patches=patches_tuple, transl_invariant=False
)

key, subkey = jax.random.split(key)
params = vit_module.init(subkey, spin_configs)

# Metropolis Local Sampling
N_samples = 4096
sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=N_samples)

optimizer = nk.optimizer.Sgd(learning_rate=0.0075)

key, subkey = jax.random.split(key, 2)
vstate = nk.vqs.MCState(
    sampler=sampler,
    model=vit_module,
    sampler_seed=subkey,
    n_samples=N_samples,
    n_discard_per_chain=0,
    variables=params,
    chunk_size=512,
)

N_params = nk.jax.tree_size(vstate.parameters)
print("Number of parameters = ", N_params, flush=True)

# Variational monte carlo driver

vmc = nk.driver.VMC_SR(
    hamiltonian=hamiltonian,
    optimizer=optimizer,
    diag_shift=1e-4,
    variational_state=vstate,
    mode="complex",
)
# %%
# Optimization
log = nk.logging.RuntimeLog()

N_opt = 800
vmc.run(n_iter=N_opt, out=log)
# %%
energy_per_site = log.data["Energy"]["Mean"].real / g_sierpinski.n_nodes

print("Last value: ", energy_per_site[-1])

plt.plot(energy_per_site)

plt.xlabel("Iterations")
plt.ylabel("Energy per site")

plt.show()
save_vstate(vstate, "vit_fractal.mpack")
# %%
