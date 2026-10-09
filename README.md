# HDG-GPU Poisson solver: FEAD reproducibility package

This archive contains the final code and saved numerical results used for the
manuscript on the GPU-accelerated hybridizable discontinuous Galerkin (HDG)
solver for the two-dimensional Poisson equation.

## Contents

- `solver/`: final runnable GPU application.
  - `main_unified_paper_sweep.py` is the definitive driver used for the data in
    manuscript Figures 6--8. It calls the HDG kernels directly; it does not call
    the historical `main.py` or `main_simple.py` entry points.
  - `paper_diagnostics.py` evaluates the residual, flux error, and local
    conservation diagnostics outside the timed interval.
  - `hdgCommon/`, `poisson/`, `referenceElement/`, `solver/`, and `Utilities/`
    contain the imported implementation modules.
  - `mesh/` contains the corrected meshes used in the study: mesh levels 1--8
    and approximation degrees p=1--4. The p=3 and p=4 files use node locations
    consistent with the corresponding reference elements.
- `postprocessing/`: saved CSV results, plotting scripts, and reference figures.
  Running `make_all_figures.py` reproduces manuscript Figures 4 and 6--8.

Exploratory drivers, obsolete diagnostic folders, Python caches, and meshes not
used in the reported sweep are intentionally excluded.

## Reference environment

The reported runs were performed on an NVIDIA TITAN V using:

- Python 3.11.16
- NumPy 2.4.6
- SciPy 1.17.1
- CuPy 14.2.0
- Numba 0.67.0
- NVIDIA driver 580.173.02

On Quijote, activate the existing `hdgenv` environment before running the
solver.

## Re-run the unified GPU sweep

From the archive root:

```bash
cd solver
python main_unified_paper_sweep.py \
  --meshes 1 2 3 4 5 6 7 8 \
  --orders 1 2 3 4 \
  --repeats 7 \
  --warmup-mesh 5 \
  --direct-threshold 1800 \
  --cg-rtol 1e-6 \
  --cg-maxiter 20000 \
  --output data/paper_unified_fp32_summary.csv
```

Each degree is warmed up once before measurement. Trace systems with at most
1800 degrees of freedom use the direct sparse solver; larger systems use
ASM-preconditioned conjugate gradients. GPU synchronization brackets the timed
pipeline. Mesh input, reference-element construction, JIT compilation, error
integration, residual evaluation, and conservation diagnostics are excluded
from the reported execution time.

## Reproduce the figures from the saved results

```bash
cd postprocessing
python make_all_figures.py
```

The generated PDF, SVG, and PNG files are written to `postprocessing/figures/`.
The postprocessing does not rerun or modify the HDG computation.

For a detailed call tree and a mapping from functions to source files, see
`postprocessing/FUNCTION_TREE.md`.
