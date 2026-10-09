# Reading the unified solver driver

`main_unified_paper_sweep.py` is preserved as the exact driver used for the
final 32-case sweep. It is intentionally not shortened here, because removing
diagnostics or moving synchronization points could change the reported study.

Read it in this order:

1. **`load_paper_mesh`** reads every degree from `codex_poisson/mesh`. The
   `p=3,4` files in that shared directory contain the corrected reference-node
   coordinates.
2. **`solve_trace_system`** contains the only global-solver decision. Small
   trace systems use `spsolve`; larger systems use ASM-preconditioned CuPy CG
   with `rtol=1e-6` and `atol=0`.
3. **`solve_case`** is one complete HDG execution:
   preprocessing, assembly and condensation, boundary projection, trace solve,
   elemental reconstruction, and postprocessing.
4. The first GPU synchronization and `start = time.perf_counter()` define the
   beginning of the measured pipeline. The second synchronization defines its
   end. Everything between them contributes to `time_pipeline`.
5. Error integration, true-residual evaluation, and local-conservation checks
   occur after the timer. They evaluate the FP32 result in FP64 arithmetic but
   do not modify the computed solution.
6. **`summarize_runs`** computes the median and 25th/75th timing percentiles for
   each `(p, mesh)` pair.
7. **`main`** performs one warm-up run per degree and then the 224 measured runs
   (4 degrees x 8 meshes x 7 repetitions).

The file writes two outputs: a per-run CSV and a summary CSV. Only the summary
CSV is read by `../plot_accuracy.py`; the per-run file is retained so every
median and quartile can be audited.
