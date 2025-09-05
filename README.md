# Operator Learning for Robust Stabilization of Linear Markov-Jumping Hyperbolic PDEs

[![License](https://img.shields.io/github/license/curryzyang/NeuralOperator4RobustStabilizationHyperbolicPDEs)](LICENSE)
![Last Commit](https://img.shields.io/github/last-commit/curryzyang/NeuralOperator4RobustStabilizationHyperbolicPDEs)
[![Issues](https://img.shields.io/github/issues/curryzyang/NeuralOperator4RobustStabilizationHyperbolicPDEs)](https://github.com/curryzyang/NeuralOperator4RobustStabilizationHyperbolicPDEs/issues)
[![Stars](https://img.shields.io/github/stars/curryzyang/NeuralOperator4RobustStabilizationHyperbolicPDEs?style=social)](https://github.com/curryzyang/NeuralOperator4RobustStabilizationHyperbolicPDEs/stargazers)

Source code of Operator Learning for Robust Stabilization of Linear Markov-Jumping Hyperbolic PDEs, together with numerical experiments (e.g., ARZ traffic model) and a CTMC simulation toolkit used for stochastic process.

## Repository Structure
- `cmtcFunc.py`: Utility functions for Continuous-Time Markov Chains (CTMC).
  - `sampleState(pvec)`, `computeNextJump(...)`, `simCTMC(...)`, `computePCFunc(...)`
  - `pltTauMatrix(...)`, `RFuncMat(...)`, `computeProbStates(...)`, `computeProbStatesFromSim(...)`
- Notebooks
  - `CMarkovChain.ipynb`: 5-state CTMC with time-varying transition rates; simulation, visualization, and state probability ODE solve.
  - `Numerical-ARZ-sto.ipynb`: Numerical experiments for the ARZ model using SciPy ODE.
  - `NO-para2k-robust.ipynb`: Neural-operator of robust stabilization experiments using PyTorch/DeepXDE and scikit-learn helpers.
  - `Plots.ipynb`: Additional plotting and data export (e.g., `scipy.io.savemat`).
  - `data_gene_train.ipynb`: Data generation.
- `requirements.txt`: Python dependencies for notebooks and scripts.
- `LICENSE`: License file.

## Installation
- Python >= 3.9
- Install dependencies

```
pip install -r requirements.txt
```

## Quick Start
### CTMC example (`CMarkovChain.ipynb`)
1. Launch Jupyter: `jupyter notebook` (or open in VS Code).
2. Run the cells sequentially. Adjust save paths to be relative to the repo (e.g., `./data/`).
3. If needed, create a `data/` folder and replace absolute paths like `/home/.../simTseq.npy` with `./data/simTseq.npy`.

### Neural Operator for robust stabilization (`NO-para2k-robust.ipynb`)
- Ensure PyTorch and DeepXDE are installed (via `requirements.txt`).

### ARZ numerical experiments (`Numerical-ARZ-sto.ipynb`)
- Run sequentially to reproduce the nominal controller for the stochastic ARZ model and plots (SciPy, Matplotlib).

## License
See the `LICENSE` file.

## Citation
If you use this repository, please cite it as follows

```
@article{zhang2024operator,
  title={Operator Learning for Robust Stabilization of Linear Markov-Jumping Hyperbolic PDEs},
  author={Zhang, Yihuai and Auriol, Jean and Yu, Huan},
  journal={arXiv preprint arXiv:2412.09019},
  year={2024}
}
```

## Contact
Feel free to leave questions in the issues of Github or email at [yzhang169@connect.hkust-gz.edu.cn](yzhang169@connect.hkust-gz.edu.cn)

