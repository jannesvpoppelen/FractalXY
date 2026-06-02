"""ViT ansatz for the XY model on fractal lattices."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import time
from functools import partial

import jax
import jax.numpy as jnp
import netket as nk
from flax import linen as nn
import optax
from einops import rearrange
from tqdm import tqdm

from utils import save_vstate, build_hamiltonian, save_metadata
from fractals import (
    honeycomb_gasket, honeycomb_gasket_coordinates,
    triangular_gasket,
)


log_cosh = nk.nn.activation.log_cosh


class Embed(nn.Module):
    d_model: int
    patches: np.ndarray
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


@partial(jax.vmap, in_axes=(None, 0, None), out_axes=1)
@partial(jax.vmap, in_axes=(None, None, 0), out_axes=1)
def roll2d(spins, i, j):
    side = int(spins.shape[-1] ** 0.5)
    spins = spins.reshape(spins.shape[0], side, side)
    spins = jnp.roll(jnp.roll(spins, i, axis=-2), j, axis=-1)
    return spins.reshape(spins.shape[0], -1)


class FMHA(nn.Module):
    d_model: int
    n_heads: int
    n_patches: int
    distances: np.ndarray
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
        self.alpha = self.param(
            "alpha",
            nn.initializers.xavier_uniform(),
            (self.n_heads, self.n_patches, self.n_patches),
            self.param_dtype,
        )
        
        # Exponent for spatial attention mechanism
        self.gamma = self.param(
            "gamma",
            lambda rng, shape: jnp.zeros(shape, dtype=self.param_dtype),
            (1,),
        )
        self.jnp_distances = jnp.array(self.distances, dtype=self.param_dtype)

    def __call__(self, x):
        v = self.v(x)
        v = rearrange(
            v,
            "batch n_patches (n_heads d_eff) -> batch n_patches n_heads d_eff",
            n_heads=self.n_heads,
        )
        v = rearrange(v, "batch n_patches n_heads d_eff -> batch n_heads n_patches d_eff")
        
        # Compute spatial attention weights based on distances
        # A_i = sum_j alpha_ij * w_ij * v_j, where w_ij = (1 + d_ij)^(-gamma)/sum_k (1 + d_ik)^(-gamma)
        w = (1 + self.jnp_distances) ** (-self.gamma)
        w = w / w.sum(axis=-1, keepdims=True)
        spatial_alpha = self.alpha * w[None, :, :]
        x = jnp.matmul(spatial_alpha[None, ...], v)
        x = rearrange(x, "batch n_heads n_patches d_eff -> batch n_patches n_heads d_eff")
        x = rearrange(x, "batch n_patches n_heads d_eff -> batch n_patches (n_heads d_eff)")
        x = self.W(x)
        return x


class EncoderBlock(nn.Module):
    d_model: int
    n_heads: int
    n_patches: int
    distances: np.ndarray
    param_dtype = jnp.float64

    def setup(self):
        self.attn = FMHA(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_patches=self.n_patches,
            distances=self.distances,
        )
        self.layer_norm_1 = nn.LayerNorm(param_dtype=self.param_dtype)
        self.layer_norm_2 = nn.LayerNorm(param_dtype=self.param_dtype)
        self.ff = nn.Sequential([
            nn.Dense(4 * self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=self.param_dtype),
            nn.gelu,
            nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=self.param_dtype),
        ])

    def __call__(self, x):
        x = x + self.attn(self.layer_norm_1(x))
        x = x + self.ff(self.layer_norm_2(x))
        return x


class Encoder(nn.Module):
    num_layers: int
    d_model: int
    n_heads: int
    n_patches: int
    distances: np.ndarray

    def setup(self):
        self.layers = [
            EncoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_patches=self.n_patches,
                distances=self.distances,
                name=f"encoder_block_{i}",
            )
            for i in range(self.num_layers)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class OuputHead(nn.Module):
    d_model: int
    param_dtype = jnp.float64

    def setup(self):
        self.out_layer_norm = nn.LayerNorm(param_dtype=self.param_dtype)
        self.norm2 = nn.LayerNorm(use_scale=True, use_bias=True, param_dtype=self.param_dtype)
        self.norm3 = nn.LayerNorm(use_scale=True, use_bias=True, param_dtype=self.param_dtype)
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
    num_layers: int
    d_model: int
    n_heads: int
    patches: np.ndarray
    distances: np.ndarray

    @nn.compact
    def __call__(self, spins):
        x = jnp.atleast_2d(spins)
        n_patches = len(self.patches)
        x = Embed(d_model=self.d_model, patches=self.patches)(x)
        y = Encoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_patches=n_patches,
            distances=self.distances,
        )(x)
        return OuputHead(d_model=self.d_model)(y)


def patch_distances(patches, x, y):
    patch_centers_x = np.array([x[patch].mean() for patch in patches])
    patch_centers_y = np.array([y[patch].mean() for patch in patches])
    return np.sqrt(
        (patch_centers_x[:, None] - patch_centers_x[None, :]) ** 2 +
        (patch_centers_y[:, None] - patch_centers_y[None, :]) ** 2
    )


def build_fractal(shape, G, n):
    """Return (g, patches, x, y) for the given fractal shape and parameters."""
    if shape == "honeycomb":
        A, patches = honeycomb_gasket(G, n)
        xa, ya, xb, yb = honeycomb_gasket_coordinates(G, n)
        x, y = np.hstack((xa, xb)), np.hstack((ya, yb))
    elif shape == "triangular":
        A, patches = triangular_gasket(G, n)
        # triangular gasket sites coincide with the A-sublattice of the honeycomb
        xa, ya, _, _ = honeycomb_gasket_coordinates(G, n)
        x, y = xa, ya
    else:
        raise ValueError(f"Unknown fractal shape '{shape}'.")

    A = A.tocoo()
    edges_list = [(i, j) for i, j in zip(A.row, A.col) if i < j]
    g = nk.graph.Graph(edges_list)
    return g, patches, x, y


def run_xy_model(
    g,
    name,
    patches,
    x,
    y,
    J=1.0,
    Jz=0.0,
    hx=0.0,
    hy=0.0,
    num_layers=4,
    d_model=48,
    n_heads=8,
    N_samples=2048,
    N_opt=800,
    n_chains=512,
    early_stop_patience=150,
    seed=0,
    outfile=None,
    fractal_config=None,
    N_symm_opt=None,
):
    start_time = time.time()

    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
    H = build_hamiltonian(hi, g, J=J, Jz=Jz, hx=hx, hy=hy)
    E, _  = nk.exact.lanczos_ed(H, compute_eigenvectors=True, k=3)
    print(f"Exact ground state energy: {E[0]}")
    distances = patch_distances(patches, x, y)

    key = jax.random.key(seed)
    patches_tuple = tuple(map(tuple, patches))
    distances_tuple = tuple(map(tuple, distances))

    vit_module = ViT(
        num_layers=num_layers,
        d_model=d_model,
        n_heads=n_heads,
        patches=patches_tuple,
        distances=distances_tuple,
    )

    key, subkey = jax.random.split(key)
    spin_configs = jax.random.randint(subkey, shape=(100, hi.size), minval=0, maxval=2) * 2 - 1

    key, subkey = jax.random.split(key)
    params = vit_module.init(subkey, spin_configs)

    sampler = nk.sampler.MetropolisLocal(hi, n_chains=n_chains)

    lr_schedule = optax.exponential_decay(
        init_value=1e-2,
        transition_steps=100,
        decay_rate=0.995,
    )
    optimizer = nk.optimizer.Sgd(learning_rate=lr_schedule)

    key, subkey = jax.random.split(key)
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
    print(f"NQS parameters: {N_params}")

    vmc = nk.driver.VMC_SR(
        hamiltonian=H,
        optimizer=optimizer,
        diag_shift=1e-4,
        variational_state=vstate,
        momentum=0.8,
        mode="complex",
    )

    energies = []
    variances = []
    best_energy = float("inf")
    patience_count = 0
    stopped_at_iter = N_opt

    with tqdm(total=N_opt, desc=name, unit="it") as pbar:
        for iteration in range(N_opt):
            vmc.advance(1)
            E = vstate.expect(H)
            energy_val = float(E.mean.real)
            variance_val = float(E.variance.real)
            energies.append(energy_val)
            variances.append(variance_val)

            pbar.set_postfix(E=f"{energy_val:.6f}", var=f"{variance_val:.2e}")
            pbar.update(1)

            if energy_val < best_energy:
                best_energy = energy_val
                patience_count = 0
            else:
                patience_count += 1

            if patience_count >= early_stop_patience and iteration > 200:
                stopped_at_iter = iteration + 1
                tqdm.write(f"Early stopping at iteration {stopped_at_iter}")
                break

    elapsed_time = time.time() - start_time
    final_energy = float(np.mean(energies[-50:]))
    final_variance = float(np.mean(variances[-50:]))
    print(f"Training time: {elapsed_time:.2f}s")

    symm_energies = []
    symm_variances = []
    proj_vstate = None

    if N_symm_opt is not None:
        group = g.automorphisms()
        rep = nk.symmetry.canonical_representation(hilbert=hi, group=group)
        proj_vstate = rep.project(vstate, character_index=0)
        proj_vstate.chunk_size = vstate.chunk_size

        symm_lr_schedule = optax.exponential_decay(
            init_value=1e-3,
            transition_steps=100,
            decay_rate=0.995,
        )
        symm_optimizer = nk.optimizer.Sgd(learning_rate=symm_lr_schedule)
        symm_vmc = nk.driver.VMC_SR(
            hamiltonian=H,
            optimizer=symm_optimizer,
            diag_shift=1e-4,
            variational_state=proj_vstate,
            momentum=0.8,
            mode="complex",
        )

        with tqdm(total=N_symm_opt, desc=f"{name}_symm", unit="it") as pbar:
            for _ in range(N_symm_opt):
                symm_vmc.advance(1)
                E = proj_vstate.expect(H)
                energy_val = float(E.mean.real)
                variance_val = float(E.variance.real)
                symm_energies.append(energy_val)
                symm_variances.append(variance_val)
                pbar.set_postfix(E=f"{energy_val:.6f}", var=f"{variance_val:.2e}")
                pbar.update(1)

        print(f"Symmetry-projected final energy: {float(np.mean(symm_energies[-50:])):.6f}")

    if outfile is not None:
        save_vstate(vstate, outfile)
        if proj_vstate is not None:
            save_vstate(proj_vstate, outfile.replace(".mpack", "_symm.mpack"))
        model_config = {
            "type": "ViT",
            "num_layers": num_layers,
            "d_model": d_model,
            "n_heads": n_heads,
        }
        sampler_config = {
            "type": "MetropolisLocal",
            "n_chains": n_chains,
            "n_samples": N_samples,
            "n_discard_per_chain": 0,
            "seed": seed,
        }
        hamiltonian_config = {"J": J, "Jz": Jz, "hx": float(hx), "hy": float(hy)}
        final_results = {
            "final_energy": final_energy,
            "final_variance": final_variance,
            "training_time": elapsed_time,
            "n_params": int(N_params),
            "stopped_at_iter": int(stopped_at_iter),
        }
        if symm_energies:
            final_results["symm_final_energy"] = float(np.mean(symm_energies[-50:]))
            final_results["symm_final_variance"] = float(np.mean(symm_variances[-50:]))
        metadata_file = outfile.replace(".mpack", "_metadata.json")
        save_metadata(
            metadata_file,
            model_config,
            sampler_config,
            hamiltonian_config,
            final_results=final_results,
            fractal=fractal_config,
        )
        print(f"Saved vstate to {outfile} and metadata to {metadata_file}")

    return {
        "vstate": vstate,
        "proj_vstate": proj_vstate,
        "hilbert": hi,
        "hamiltonian": H,
        "energies": np.array(energies),
        "variances": np.array(variances),
        "final_energy": final_energy,
        "final_variance": final_variance,
        "symm_energies": np.array(symm_energies),
        "symm_variances": np.array(symm_variances),
        "n_params": N_params,
        "stopped_at_iter": stopped_at_iter,
        "training_time": elapsed_time,
    }


if __name__ == "__main__":
    print(f"NetKet {nk.__version__}, Jax {jax.__version__}")
    print(f"JAX is using: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")


    shapes = ["honeycomb"]
    generations = [1]
    n = 2
    seeds = [1]

    for shape in shapes:
        for G in generations:
                g, patches, x, y = build_fractal(shape, G, n)
                print(f"\n{'='*60}")
                print(f"{shape} gasket G={G}: {g.n_nodes} nodes")
                print(f"{'='*60}")

                os.makedirs("data/ViT", exist_ok=True)
                for seed in seeds:
                    name = f"vit_{shape}_G={G}_seed={seed}"
                    run_xy_model(
                        g=g,
                        name=name,
                        patches=patches,
                        x=x,
                        y=y,
                        J=1.0,
                        Jz=0.0,
                        hx=0.0,
                        hy=0.0,
                        num_layers=3,
                        d_model=40,
                        n_heads=5,
                        N_samples=1024,
                        N_opt=500,
                        N_symm_opt=100,
                        n_chains=1024,
                        early_stop_patience=150,
                        seed=seed,
                        outfile=f"data/ViT/{name}.mpack",
                        fractal_config={"shape": shape, "G": G, "n": n},
                    )
