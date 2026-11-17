# Quantum XY model on Sierpinski gaskets

Code and data for *Quantum XY model on the Sierpinski triangle* (arXiv link to follow).

We study the spin-1/2 XY model on triangular and honeycomb Sierpinski gaskets with neural quantum states (NetKet). The ground state is approximated by a vision transformer (ViT) trained with VMC and minSR, followed by a $C_{3v}$ symmetry projection.

The final results come from a second stage: each state is retrained inside its fixed $S^z$ sector (the ground-state sector, $S^z = 0$ for an even number of sites and $+1/2$ for an odd one), which the full-space training does not enforce. These are the figures made by `pp_sz.py`.

Questions: [jannes.vanpoppelen@physics.uu.se](mailto:jannes.vanpoppelen@physics.uu.se)

## Requirements

Python 3.11–3.12 with `netket`, `jax`, `flax`, `optax`, `einops`, `numpy`, `scipy`, `networkx` and `tqdm`. Plotting additionally needs `matplotlib`, `Pillow` and `pdftoppm` (poppler); `scienceplots` is used if installed.

The $S^z$-sector runs used Python 3.11 with netket 3.21.0, jax 0.8.1, flax 0.12.2, optax 0.2.6, einops 0.8.1, numpy 2.3.1, scipy 1.17.1, networkx 3.6.1 and matplotlib 3.10.7.

## Files

| file | |
|---|---|
| `fractals.py` | adjacency matrices, coordinates and patches of the gaskets and of the clean reference lattices |
| `utils.py` | Hamiltonian, I/O and distance shells |
| `trainViT.py` | ViT ansatz and training (main results) |
| `trainRBM.py`, `trainGCNN.py` | RBM and GCNN ansätze |
| `hyperSearchViT.py` | one point of the ViT hyperparameter grid search |
| `expect.py` | loading saved states and computing observables (energy, magnetization, $C_{xy}$ correlators) |
| `fidelity.py` | pairwise fidelities between independently trained states |
| `run_rbm_bench.py` | RBM benchmark on gaskets vs. clean lattices, including ED for small systems |
| `pp.py` | figures of the first version (full-space states); `pp_sz.py` reuses its plotting functions |
| **$S^z$ sector** | |
| `trainViT_sz.py` | retraining in the fixed $S^z$ sector, warm-started from the `data/ViT` states |
| `sector_fidelity.py` | loading sector states, sector ED, and seed-to-seed fidelities inside the sector |
| `ed_sector.py` | exact (Lanczos) lowest energies of a sector for small gaskets, written to `data/ED/` |
| `promote_ckpt.py` | makes the last checkpoint of a `trainViT_sz.py` run its saved state again |
| `pp_sz.py` | **the figures of the paper**: fig1, fig2, fig3_sz_vscore, fig4_sz, fig5_sz |
| `compare_correlations.py`, `fit_correlations.py`, `degeneracy.py`, `plot_training.py` | checks: original vs. retrained correlations, weighted fits, seed degeneracy, training curves |

## Reproducing

Stage 1, full space (first version):

```
python trainViT.py       # trains all shapes, generations and seeds
python expect.py         # observables of the trained states
python fidelity.py       # seed-to-seed fidelities
python pp.py             # figures of the first version
```

Training was done on a single GPU per (shape, G, seed); a G=4 run takes roughly 1.7 h on an A100.

Stage 2, $S^z$ sector (paper figures):

```
python trainViT_sz.py --n-opt 400 --c3v-steps 20 --eval-discard 256
python expect.py --sector data/ViT_sz          # observables of the retrained states
python sector_fidelity.py --source data/ViT_sz # fidelities inside the sector
python run_rbm_bench.py                         # clean-lattice reference in fig4_sz
python pp_sz.py                                 # paper figures
```
## Data

`data/ViT/<shape>/` contains, for every generation `G` and seed:

- `vit_<shape>_G=<G>_seed=<s>.mpack`: trained state (flax serialization), `_symm.mpack` after symmetry projection
- `..._metadata.json`: model, sampler and Hamiltonian settings and final energy/variance; needed to load the `.mpack`
- `..._energy_history.json`: energy and variance during training
- `..._observables.json`, `..._symm_observables.json`: observables before and after projection
- `fidelity_<shape>_gasket_G=<G>[_symm].json`: pairwise fidelities

`data/ViT_sz/<shape>/` contains the states retrained in the $S^z$ sector, with the same file types:

- `vit_<shape>_G=<G>_seed=<s>_symm.mpack` and its `_metadata.json`, `_energy_history.json`, `_observables.json`
- `..._symm_prev*`: first round of the runs continued with `--continue`
- `fidelity_<shape>_gasket_G=<G>_sector_symm.json`: pairwise fidelities inside the sector

`data/ED/` holds the exact sector energies of the small gaskets, `data/RBM_bench/` the RBM benchmark, and `data/ViT_hypersearchG3/`, `data/ViT_hypersearchG4/` the hyperparameter search.

The `.mpack` network states are not in the git repository; they are in the Zenodo record.

To load a state, see `load_vstate_vit` and `load_vstate_vit_symm` in `expect.py` (full space) and `load_retrained_state` in `sector_fidelity.py` ($S^z$ sector).
