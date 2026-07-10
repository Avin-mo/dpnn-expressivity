# dpnn-expressivity

Experiments on neural network expressivity, training dynamics, and parameter-space
geometry, run in Dr. Maksym Zubkov's lab at UBC. The question underlying all three
notebooks is: for a small, fixed architecture, does gradient descent actually find the
parameters that realize a given target function, and if not, why not.

Each notebook trains a small two-layer network (`SimpleDPN`) on a fixed target function
and architecture, sweeps many independent training runs, and analyzes the resulting
weights to look for structure in how (or whether) the network converges.

## Notebooks

| Notebook | Architecture `d=(in, hidden, out)` | Target function |
|---|---|---|
| `dpnn_one_input.ipynb` | `(1, 2, 1)` | $f(x) = \frac{1}{1+(x-1)^2} + \frac{1}{1+(x+1)^2}$, sum of two shifted Lorentzians |
| `dpnn_three_input.ipynb` | `(3, 3, 1)` | $f(x,y,z) = x^2 + y^2 - z^2$, a cone |
| `dpnn_four_inputs.ipynb` | `(4, 2, 1)` | $f(x,y,z) = x^2 + y^2 - z^2$, same cone, lower hidden width |

`dpnn_one_input.ipynb` is the active notebook and the one referenced in current research
notes. The other two are earlier higher-dimensional variants of the same pipeline.

## What each notebook does

1. **Model definition.** `SimpleDPN`: a two-layer fully connected network. The hidden
   layer is followed by either a polynomial activation ($x \mapsto x^p$) or a custom
   rational (Lorentzian-style) activation, $x \mapsto \frac{1}{1+x^2} + \text{bias}$, with
   the bias itself a learnable parameter.
2. **Data.** Training and test sets are regular lattices sampled from the target function
   at different resolutions.
3. **Single-run training.** Trains one network with Adam + MSE loss, plots the loss curve,
   and evaluates on the held-out lattice.
4. **N-run sweep.** Repeats training from N independent random initializations (40 runs at
   10,000 epochs each in the current setup), saving each run's initial weights, final
   weights, and loss curve to disk.
5. **Analysis.** Reloads the saved runs and:
   - filters runs by final loss to separate networks that converged from ones that didn't
   - projects flattened weight vectors with PCA to look for clustering by outcome
   - computes eigenvalues of $W_1 W_1^T$ per run
   - inspects individual runs' weights directly

## Why this exists

The core question is why training loss fails to reach zero even though the architecture
should have enough capacity to represent the target exactly. Two findings so far, from the
`(1,2,1)` / Lorentzian setup:

- With no bias on the first linear layer, every hidden unit's activation is forced to peak
  at $x=0$, which constrains what the network can represent regardless of training length.
- Even after fixing that, some runs still show one hidden neuron collapsing to near-zero
  weights ("neuron death"), effectively reducing a `(1,2,1)` network to `(1,1,1)`. Whether
  this comes from initialization or from gradient dynamics during training is the current
  open question, being investigated via the weight and eigenvalue analysis in Section 7.

## Requirements
- torch
- numpy
- scikit-learn
- matplotlib
- plotly

## Usage
Each notebook is self-contained and organized into numbered sections (imports, model
definition, data, single-run training, N-run sweep, analysis). Section 5 (the sweep) is
the expensive step; Sections 6-7 can be re-run on their own against saved `.npy` files
without repeating training.

## Acknowledgments
Parts of the training and analysis pipeline (the model definition, sweep/save
infrastructure, and eigenvalue analysis in Section 7) build on code originally written
by Dr. Maksym Zubkov, whose lab this research is conducted in.
