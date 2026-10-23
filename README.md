# Quantum XY model on Sierpinski gaskets

Code and data for *Quantum XY model on the Sierpinski triangle* (arXiv link to follow).

We study the spin-1/2 XY model on triangular and honeycomb Sierpinski gaskets with neural quantum states (NetKet). The ground state is approximated by a vision transformer (ViT) trained with VMC and minSR, followed by a $C_{3v}$ symmetry projection.

Questions: [jannes.vanpoppelen@physics.uu.se](mailto:jannes.vanpoppelen@physics.uu.se)

## Requirements

Python 3.12 with `netket`, `jax`, `flax`, `optax`, `einops`, `numpy`, `scipy`, `networkx` and `tqdm`. Plotting additionally needs `matplotlib`, `scienceplots`, `Pillow` and `pdftoppm` (poppler).

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
| `pp.py` | all figures of the paper |

## Reproducing

```
python trainViT.py       # trains all shapes, generations and seeds
python expect.py         # observables of the trained states
python fidelity.py       # seed-to-seed fidelities
python pp.py             # figures fig1-fig6
```

Training was done on a single GPU per (shape, G, seed); a G=4 run takes roughly 1.7 h on an A100.

## Data

`data/ViT/<shape>/` contains, for every generation `G` and seed:

- `vit_<shape>_G=<G>_seed=<s>.mpack`: trained state (flax serialization), `_symm.mpack` after symmetry projection
- `..._metadata.json`: model, sampler and Hamiltonian settings and final energy/variance; needed to load the `.mpack`
- `..._energy_history.json`: energy and variance during training
- `..._observables.json`, `..._symm_observables.json`: observables before and after projection
- `fidelity_<shape>_gasket_G=<G>[_symm].json`: pairwise fidelities

`data/RBM_bench/` holds the RBM benchmark, and `data/ViT_hypersearchG3/`, `data/ViT_hypersearchG4/` the hyperparameter search.

To load a state, see `load_vstate_vit` and `load_vstate_vit_symm` in `expect.py`.
