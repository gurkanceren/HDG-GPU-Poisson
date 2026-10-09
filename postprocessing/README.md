# Final paper figure workflow

This folder contains the shortest reproducible path from the saved numerical
results to manuscript Figures 4 and 6--8.

## Which code is final?

- `solver/main_unified_paper_sweep.py` is the exact unified GPU driver used to
  generate the data for Figures 6--8. It must be run on Quijote from the root of
  the HDG `codes` directory because it imports the solver packages and requires
  CuPy/CUDA.
- `plot_accuracy.py` creates Figures 6--8. This is the best file to read first.
- `plot_timing_breakdown.py` creates Figure 4 and contains its more elaborate
  stacked-bar and grouped-legend formatting.
- `make_all_figures.py` is a small launcher that runs both plotting files.
- `data/final_unified_all_runs.csv` stores all 224 timed runs: 8 meshes, 4
  polynomial degrees, and 7 repetitions.
- `data/final_unified_summary.csv` stores the medians, timing quartiles, errors,
  residuals, and conservation diagnostics used by Figures 6--8.
- `data/final_v16_stage_timings.csv` stores the v1.6 stage timings used by
  Figure 4.

The plotting script does not rerun or modify the HDG solution. It only reads
the saved CSV files.

On Quijote, all polynomial degrees are now loaded from
`codex_poisson/mesh`. Its `p=3,4` files use the corrected reference-element
node distribution; the same corrected files are installed in
`refactored/mesh_files`. Daniel's historical `codes/mesh` directory is retained
unchanged.

## Recreate the figures

From this folder, run:

```bash
python make_all_figures.py
```

This creates PDF, SVG, and PNG versions under `figures/`:

- `V6_absolutetimings`: manuscript Figure 4;
- `time_to_accuracy_fp32`: manuscript Figure 6;
- `dof_to_accuracy_fp32`: manuscript Figure 7;
- `time_accuracy_pareto_fp32`: manuscript Figure 8.

Required Python packages are NumPy and Matplotlib.

The reference files currently stored in `figures/` were rendered with Quijote's
`hdgenv` and match the manuscript figures pixel-for-pixel. Running the same
script with a different Matplotlib or font installation can change glyph
antialiasing, but not the plotted data, minima, or Pareto membership.

## How Figures 6--8 are constructed

1. `load_results()` reads one row for every `(p, mesh)` combination.
2. `one_degree()` sorts the eight mesh levels from coarse to fine.
3. `draw_degree_curves()` finds the smallest measured error for each degree.
   Markers through that minimum are filled; later refinements are open.
4. `make_curve_figure()` draws error against either median GPU
   pipeline time (Figure 6) or trace degrees of freedom (Figure 7).
5. `pareto_frontier()` sorts all cases by time and retains a point only when it
   improves the smallest error reached by every faster point (Figure 8).

The horizontal bars in Figure 6 are the 25th-to-75th percentile range of the
seven synchronized timings. They are timing uncertainty bars, not error bars for
the numerical solution.

## How Figure 4 is constructed

`plot_timing_breakdown.main()` draws one stacked bar for every mesh and degree. Each
colored segment is one column of `final_v16_stage_timings.csv`. The printed
number above a bar is the saved total execution time, converted to milliseconds.
The script independently checks that the sum of the stage columns agrees with
the saved total.

## Re-run the unified GPU sweep

On Quijote, enter the retained `codex_poisson` application and run:

```bash
cd ~/playground_Daniel_clean/codex_poisson
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

The warm-up executes each polynomial degree once before timing. Systems with at
most 1800 total trace degrees of freedom use the direct sparse solver; larger
systems use ASM-preconditioned CG. The timed interval is synchronized on the
GPU and excludes mesh-file input, reference-element construction, compilation,
error integration, residual evaluation, and conservation diagnostics.
