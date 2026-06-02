import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax
import jax.random as jr
import netket as nk
from flax import linen as nn
import time

from utils import *
from fractals import triangular_gasket, honeycomb_gasket

print(f"NetKet {nk.__version__}, Jax {jax.__version__}")
print(f"JAX is using: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

G=1
n=2
A, _ = triangular_gasket(G, n)
A = A.tocoo()   
edges = list(zip(A.row, A.col))
edges_list = [(i, j) for i, j in edges if i < j]
g = nk.graph.Graph(edges_list)

hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
H = build_hamiltonian(hi, g, 1, 0, 0, 0)

group = g.automorphisms()
rep = nk.symmetry.canonical_representation(
    hilbert=hi,
    group=group)

model = nk.models.RBM(alpha=1, kernel_init=nn.initializers.normal(stddev=0.01), hidden_bias_init=nn.initializers.normal(stddev=0.01))
sampler = nk.sampler.MetropolisLocal(hi, n_chains=1)
vstate = nk.vqs.MCState(sampler, model, n_samples=1024, n_discard_per_chain= 2**7, seed=0)
proj_vstate = rep.project(vstate, character_index = 0)