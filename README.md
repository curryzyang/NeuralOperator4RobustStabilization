# NeuralOperator4RobustStabilizationHyperbolicPDEs

[![License](https://img.shields.io/github/license/curryzyang/NeuralOperator4RobustStabilizationHyperbolicPDEs)](LICENSE)
![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue)
![Last Commit](https://img.shields.io/github/last-commit/curryzyang/NeuralOperator4RobustStabilizationHyperbolicPDEs)
[![Issues](https://img.shields.io/github/issues/curryzyang/NeuralOperator4RobustStabilizationHyperbolicPDEs)](https://github.com/curryzyang/NeuralOperator4RobustStabilizationHyperbolicPDEs/issues)
[![Stars](https://img.shields.io/github/stars/curryzyang/NeuralOperator4RobustStabilizationHyperbolicPDEs?style=social)](https://github.com/curryzyang/NeuralOperator4RobustStabilizationHyperbolicPDEs/stargazers)

A small collection of notebooks and utilities exploring neural-operator-based modeling/training under nominal settings for hyperbolic PDEs, together with numerical experiments (e.g., ARZ traffic model) and a CTMC simulation toolkit used for stochastic modeling/verification.

## Overview
An overview illustration can be placed at `docs/figs/overview.svg` (already present) or `docs/figs/overview.png`.

![Overview](docs/figs/overview.svg)

## Features
- CTMC simulation utilities (sampling, jump times, piecewise-constant reconstruction, Kolmogorov equations, plotting).
- Numerical experiments for the ARZ model with sparse/ODE solvers.
- A notebook for neural-operator-style training under nominal mode, relying on PyTorch and DeepXDE utilities.
- Lightweight plotting and data-saving helpers in notebooks.

## Repository Structure
- `cmtcFunc.py`: Utility functions for Continuous-Time Markov Chains (CTMC).
  - `sampleState(pvec)`, `computeNextJump(...)`, `simCTMC(...)`, `computePCFunc(...)`
  - `pltTauMatrix(...)`, `RFuncMat(...)`, `computeProbStates(...)`, `computeProbStatesFromSim(...)`
- Notebooks
  - `CMarkovChain.ipynb`: 5-state CTMC with time-varying transition rates; simulation, visualization, and state probability ODE solve.
  - `Numerical-ARZ-sto.ipynb`: Numerical experiments for the ARZ model using SciPy ODE/sparse routines.
  - `NO-para2k-nominalmode.ipynb`: Neural-operator/nominal-mode experiments using PyTorch/DeepXDE and scikit-learn helpers.
  - `Plots.ipynb`: Additional plotting and data export (e.g., `scipy.io.savemat`).
  - `data_gene_train.ipynb`: Data generation/training workflow.
- `docs/figs/overview.svg`: Overview figure.
- `requirements.txt`: Python dependencies for notebooks and scripts.
- `LICENSE`: License file.

## Installation
- Python >= 3.9
- Install dependencies

```
pip install -r requirements.txt
```

Notes
- The notebooks import `utilities3` (e.g., `from utilities3 import LpLoss`). This is not a pip package. Place `utilities3.py` at the project root (e.g., copy from the Fourier Neural Operator repo) or ensure it is on `PYTHONPATH`.
- For Apple Silicon (macOS), PyTorch uses Metal (MPS). Recent PyTorch releases enable `mps` by default when available.

## Quick Start
### CTMC example (`CMarkovChain.ipynb`)
1. Launch Jupyter: `jupyter notebook` (or open in VS Code).
2. Run the cells sequentially. Adjust save paths to be relative to the repo (e.g., `./data/`).
3. If needed, create a `data/` folder and replace absolute paths like `/home/.../simTseq.npy` with `./data/simTseq.npy`.

### Neural Operator nominal mode (`NO-para2k-nominalmode.ipynb`)
- Ensure PyTorch and DeepXDE are installed (via `requirements.txt`).
- Provide `utilities3.py` as noted above.
- Select a device (`cuda`, `mps`, or `cpu`) if the notebook exposes a device selector. Then run cells.

### ARZ numerical experiments (`Numerical-ARZ-sto.ipynb`)
- Run sequentially to reproduce ODE/sparse-based computations and plots (SciPy, Matplotlib).

## Tips
- Reproducibility: set a random seed, e.g. `np.random.seed(0)` and (for PyTorch) `torch.manual_seed(0)`.
- Parallelism: `computeProbStatesFromSim` uses `multiprocess`; control concurrency with `nProcesses`.
- Figures: save to `docs/figs/` for inclusion in the README.

## License
See the `LICENSE` file.

## Citation
If you use this repository, please cite it as follows

```
@misc{NeuralOperator4RobustStabilizationHyperbolicPDEs,
  title        = {NeuralOperator4RobustStabilizationHyperbolicPDEs},
  author       = {curryzyang},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/curryzyang/NeuralOperator4RobustStabilizationHyperbolicPDEs}}
}
```

## Acknowledgment
Thanks to the open-source community (NumPy, SciPy, Matplotlib, PyTorch, DeepXDE) and related operator-learning resources.
