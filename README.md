# Investigation of Quantum XY model in Sierpinksi Triangles

Code repository for [Quantum XY model on Sierpinski triangle](). The repository contains the code, and data, to generate the figures and run the experiments presented in the paper. For any questions, please contact via email [jannes.vanpoppelen@physics.uu.se](mailto:jannes.vanpoppelen@physics.uu.se)

To generate the vertices and edges for the Sierpinski triangle, run:

```julia fractal.jl```

To train the NQS, run:

```python train.py```

Per generation of fractal and random seed, this saves the NQS, as well as the metadata necessary to correctly load the saved NQS.


To compute expectation values, run:

```python expect.py```

To compute pairwise fidelities, run:

```python fidelity.py```

To generate the figures, run:

```python plot.py```

Make sure the paths to the data is correct.
