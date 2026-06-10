"""Quick smoke test: load a saved ViT vstate and compute expectation values."""

import os

import numpy as np
import netket as nk
from netket.operator.spin import sigmax, sigmay, sigmaz

from trainViT import build_fractal, patch_distances
from expect import load_vstate_vit, load_vstate_vit_symm
from utils import build_hamiltonian

SHAPE = "honeycomb"
G = 2
N = 2
SEED = 1

VSTATE_FILE = f"data/ViT/vit_{SHAPE}_G={G}_seed={SEED}_symm.mpack"
METADATA_FILE = f"data/ViT/vit_{SHAPE}_G={G}_seed={SEED}_metadata.json"

print(f"Building {SHAPE} gasket G={G}, n={N}...")
g, patches, x, y = build_fractal(SHAPE, G, N)
print(f"  {g.n_nodes} sites, {len(patches)} patches")

distances = patch_distances(patches, x, y)
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)

print(f"\nLoading vstate from {VSTATE_FILE}...")
vstate = load_vstate_vit_symm(VSTATE_FILE, METADATA_FILE, hi, g, patches, distances)
vstate.n_samples = 4096
print("  Loaded OK")

H = build_hamiltonian(hi, g, J=1.0, Jz=0.0)
Mx = sum(sigmax(hi, i) for i in range(g.n_nodes)) / g.n_nodes
My = sum(sigmay(hi, i) for i in range(g.n_nodes)) / g.n_nodes
Mz = sum(sigmaz(hi, i) for i in range(g.n_nodes)) / g.n_nodes

print("\nComputing observables...")
E   = vstate.expect(H)
mx  = vstate.expect(Mx)
my  = vstate.expect(My)
mz  = vstate.expect(Mz)

print(f"  E   = {E.mean.real:.6f} ± {E.error_of_mean.real:.2e}  (var={E.variance.real:.2e})")
print(f"  Mx  = {mx.mean.real:.6f} ± {mx.error_of_mean.real:.2e}")
print(f"  My  = {my.mean.real:.6f} ± {my.error_of_mean.real:.2e}")
print(f"  Mz  = {mz.mean.real:.6f} ± {mz.error_of_mean.real:.2e}")

print("\nDone.")
