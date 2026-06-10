"""
Physics tests for symmetry projection onto the trivial irrep (character_index=0).
Uses the G=2 honeycomb gasket (~12 nodes), which is small enough for exact diagonalization.

Run with:  python test_projection.py
"""

import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
from netket.operator.spin import sigmax, sigmay

from utils import build_hamiltonian
from fractals import honeycomb_gasket, honeycomb_gasket_coordinates
from trainViT import ViT, patch_distances, build_fractal

# -----------------------------------------------------------------------
# Setup: G=2 honeycomb gasket
# -----------------------------------------------------------------------

G, n = 2, 2
g, patches, x, y = build_fractal("honeycomb", G, n)
print(f"Honeycomb gasket G={G}: {g.n_nodes} nodes, {len(list(g.edges()))} edges")

hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
H  = build_hamiltonian(hi, g, J=1.0, Jz=0.0)

distances     = patch_distances(patches, x, y)
patches_tuple = tuple(map(tuple, patches))
distances_tuple = tuple(map(tuple, distances))

# Small ViT — enough to get a reasonable (not necessarily converged) state
vit_module = ViT(num_layers=2, d_model=16, n_heads=4,
                 patches=patches_tuple, distances=distances_tuple)

key = jax.random.key(42)
spin_configs = jax.random.randint(key, shape=(64, hi.size), minval=0, maxval=2) * 2 - 1
key, subkey = jax.random.split(key)
params = vit_module.init(subkey, spin_configs)

sampler = nk.sampler.MetropolisLocal(hi, n_chains=64)
key, subkey = jax.random.split(key)
vstate = nk.vqs.MCState(sampler=sampler, model=vit_module,
                         sampler_seed=subkey, n_samples=512,
                         variables=params, chunk_size=64)

# Brief optimisation so the state is non-trivial
optimizer = nk.optimizer.Sgd(learning_rate=1e-2)
vmc = nk.driver.VMC_SR(hamiltonian=H, optimizer=optimizer, diag_shift=1e-4,
                        variational_state=vstate, momentum=0.8, mode="complex")
print("Running 50 VMC steps …")
for _ in range(250):
    vmc.advance(1)

E_before = vstate.expect(H)
energy_before   = float(E_before.mean.real)
variance_before = float(E_before.variance.real)
print(f"\nBefore projection — E = {energy_before:.6f},  var = {variance_before:.2e}")

# -----------------------------------------------------------------------
# Project onto trivial symmetry sector
# -----------------------------------------------------------------------

group = g.automorphisms()
rep   = nk.symmetry.canonical_representation(hilbert=hi, group=group)
print(f"Automorphism group order: {len(group)}")

proj_vstate = rep.project(vstate, character_index=0)
proj_vstate.chunk_size = vstate.chunk_size

E_after = proj_vstate.expect(H)
energy_after   = float(E_after.mean.real)
variance_after = float(E_after.variance.real)
print(f"After  projection — E = {energy_after:.6f},  var = {variance_after:.2e}")

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

# -----------------------------------------------------------------------
# Test 1: Energy should not increase after projection
# -----------------------------------------------------------------------

tol_energy = 3 * (E_before.error_of_mean + E_after.error_of_mean)  # 3-sigma
delta_E = energy_after - energy_before
result = PASS if delta_E <= tol_energy else FAIL
print(f"\n[Test 1] Energy non-increase:  ΔE = {delta_E:+.6f}  (tol = {tol_energy:.4f})  {result}")

# -----------------------------------------------------------------------
# Test 2: Variance should decrease after projection
# -----------------------------------------------------------------------

result = PASS if variance_after < variance_before else FAIL
print(f"[Test 2] Variance decrease:    {variance_before:.2e} → {variance_after:.2e}  {result}")

# -----------------------------------------------------------------------
# Test 3: Symmetry-related bond correlators must be equal
#
# Pick the first edge and find another edge related to it by an automorphism.
# Their XY correlators <σᵢˣσⱼˣ + σᵢʸσⱼʸ> must match in the projected state.
# -----------------------------------------------------------------------

edges = list(g.edges())
i0, j0 = edges[0]

# Find a symmetry-related edge (one that is NOT the same pair)
perm_array = np.array(group.to_array())   # shape (n_symm, n_sites)
ref_set = frozenset((i0, j0))
partner = None
for perm in perm_array:
    pi, pj = int(perm[i0]), int(perm[j0])
    candidate = frozenset((pi, pj))
    if candidate != ref_set and candidate in {frozenset(e) for e in edges}:
        partner = (pi, pj)
        break

if partner is None:
    print("[Test 3] Bond symmetry:        could not find a symmetry-related edge — SKIP")
else:
    i1, j1 = partner
    C_ref = sigmax(hi, i0) @ sigmax(hi, j0) + sigmay(hi, i0) @ sigmay(hi, j0)
    C_par = sigmax(hi, i1) @ sigmax(hi, j1) + sigmay(hi, i1) @ sigmay(hi, j1)

    val_ref = float(proj_vstate.expect(C_ref).mean.real)
    val_par = float(proj_vstate.expect(C_par).mean.real)
    err_ref = float(proj_vstate.expect(C_ref).error_of_mean)
    err_par = float(proj_vstate.expect(C_par).error_of_mean)

    tol_corr = 3 * (err_ref + err_par)
    result = PASS if abs(val_ref - val_par) < tol_corr else FAIL
    print(f"[Test 3] Bond symmetry:        C({i0},{j0}) = {val_ref:.4f},  "
          f"C({i1},{j1}) = {val_par:.4f},  |Δ| = {abs(val_ref-val_par):.4f}  (tol = {tol_corr:.4f})  {result}")

# -----------------------------------------------------------------------
# Test 4: Idempotency — projecting twice should give the same energy
# -----------------------------------------------------------------------

rep2          = nk.symmetry.canonical_representation(hilbert=hi, group=group)
proj2_vstate  = rep2.project(proj_vstate, character_index=0)
E_proj2       = proj2_vstate.expect(H)
energy_proj2  = float(E_proj2.mean.real)

tol_idem = 3 * (E_after.error_of_mean + E_proj2.error_of_mean)
result = PASS if abs(energy_after - energy_proj2) < tol_idem else FAIL
print(f"[Test 4] Idempotency:          E(P) = {energy_after:.6f},  E(P²) = {energy_proj2:.6f},  "
      f"|Δ| = {abs(energy_after - energy_proj2):.2e}  (tol = {tol_idem:.2e})  {result}")

# # -----------------------------------------------------------------------
# # Test 5: Exact diagonalization comparison
# # -----------------------------------------------------------------------

# print("\nRunning exact diagonalization …")
# E_exact = nk.exact.lanczos_ed(H, compute_eigenvectors=False)
# print(f"ED ground-state energy: {E_exact:.6f}")
# print(f"VMC projected energy:   {energy_after:.6f}  ±  {float(E_after.error_of_mean):.4f}")

# gap = energy_after - E_exact
# tol_ed = 3 * float(E_after.error_of_mean)
# result = PASS if gap < 0.05 * abs(E_exact) else "CLOSE" if gap < 0.10 * abs(E_exact) else FAIL
# print(f"[Test 5] ED comparison:        gap = {gap:.4f}  ({100*gap/abs(E_exact):.1f}% above ED)  {result}")
# print("         (gap > 0 is expected for a partially trained state; PASS = within 5% of ED)")

# print("\nDone.")
