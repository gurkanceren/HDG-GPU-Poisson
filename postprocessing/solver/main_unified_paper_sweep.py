"""Unified FP32 HDG sweep for the manuscript time-to-accuracy study.

Every reported time, error, iteration count, residual, and conservation defect
is obtained from the same execution.  The solver uses the direct sparse solve
for small condensed systems and ASM-preconditioned CG for larger systems.  FP64
is used only after the solve for diagnostic integration and residual evaluation;
it never changes the FP32 matrices or solution fields.

This file is intended to be copied into the root of the Quijote ``codes``
directory.  The accompanying analytical/source and solver patch files in this
folder must also be installed in their documented package locations.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cupy as cp
import numpy as np
from cupyx.scipy.sparse.linalg import cg, spsolve

from paper_diagnostics import compute_flux_l2_error, compute_local_conservation
from hdgCommon.hdg_preprocess import hdg_preprocess
from poisson.analyticalPoisson import analytical_poisson
from referenceElement.createReferenceElementTri import create_reference_element_tri
from referenceElement.createReferenceElementTriStar import (
    create_reference_element_tri_star,
)
from solver.compute_elements_solution_gpu import compute_elements_solution_gpu
from solver.compute_projection_faces import compute_projection_faces_gpu
from solver.dense_blocks.matvec import linearOperator_RawKernel
from solver.matrix_assembly import matrix_assembly
from solver.post_processing_new import HDG_postprocess_gpu
from solver.preconditioners.additive_schwarz import create_ASM_preconditioner
from solver.tracker import CGTracker
from Utilities.computeL2Norm import compute_L2_norm
from Utilities.computeL2NormPostProcessed import compute_L2_norm_postprocess


FLOAT_DTYPE = cp.float32
INT_DTYPE = cp.int32
N_FACES_PER_ELEMENT = 3
N_COMPONENTS = 1
MU = 1.0


def as_numpy(array):
    """Return a host array without asking CuPy for an implicit conversion."""
    if isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def load_paper_mesh(mesh_number, order, mesh_root):
    """Load a mesh; the shared directory contains corrected p=3,4 nodes."""
    mesh_path = mesh_root / f"mesh{mesh_number}_P{order}.npz"
    if not mesh_path.is_file():
        raise FileNotFoundError(
            f"Required mesh not found: {mesh_path}. "
            "For p=3 and p=4, use the corrected reference-node meshes."
        )

    with np.load(mesh_path) as mesh:
        nodes = mesh["NODES"]
        connectivity = mesh["CONN"]
        dirichlet_connectivity = mesh["Tb_Dirichlet"]
    return nodes, connectivity, dirichlet_connectivity, mesh_path


def evaluate_true_residual(K_uu_gpu, solution_gpu, rhs_gpu):
    """Evaluate the same FP32 solution with FP32 and FP64 residual arithmetic."""
    residual32 = K_uu_gpu.dot(solution_gpu) - rhs_gpu
    rhs_norm32 = cp.linalg.norm(rhs_gpu)
    relative32 = cp.linalg.norm(residual32) / rhs_norm32

    K64 = K_uu_gpu.astype(cp.float64)
    x64 = solution_gpu.astype(cp.float64)
    b64 = rhs_gpu.astype(cp.float64)
    relative64 = cp.linalg.norm(K64.dot(x64) - b64) / cp.linalg.norm(b64)
    return float(relative32.get()), float(relative64.get())


def solve_trace_system(
    K_uu_gpu,
    f_reduced_gpu,
    Ke,
    fe,
    faces_gpu,
    interior_faces_gpu,
    exterior_faces_gpu,
    face_neighbors_gpu,
    n_elements,
    n_nodes_per_face,
    n_interior_faces,
    n_dof_total,
    direct_threshold,
    cg_rtol,
    cg_maxiter,
):
    """Solve once using the manuscript's direct/ASM-CG selection rule."""
    K_uu_gpu = K_uu_gpu.tocsr()

    if n_dof_total <= direct_threshold:
        solution_gpu = spsolve(K_uu_gpu, f_reduced_gpu)
        return solution_gpu, "direct", 0, 0

    asm_gpu = create_ASM_preconditioner(
        Ke,
        fe,
        faces_gpu,
        interior_faces_gpu,
        exterior_faces_gpu,
        n_elements,
        n_nodes_per_face,
        N_FACES_PER_ELEMENT,
        n_interior_faces,
    )
    asm_reduced_gpu = asm_gpu[: n_interior_faces * n_nodes_per_face,
                              : n_interior_faces * n_nodes_per_face]

    matrix_tracker = CGTracker()
    preconditioner_tracker = CGTracker()
    n_neighbors = 5
    matrix_operator = linearOperator_RawKernel(
        K_uu_gpu,
        face_neighbors_gpu,
        n_interior_faces,
        n_nodes_per_face,
        n_neighbors,
        matrix_tracker,
    )
    preconditioner_operator = linearOperator_RawKernel(
        asm_reduced_gpu,
        face_neighbors_gpu,
        n_interior_faces,
        n_nodes_per_face,
        n_neighbors,
        preconditioner_tracker,
    )

    iteration_count = [0]

    def count_iteration(_):
        iteration_count[0] += 1

    solution_gpu, info = cg(
        matrix_operator,
        f_reduced_gpu,
        M=preconditioner_operator,
        rtol=cg_rtol,
        atol=0.0,
        maxiter=cg_maxiter,
        callback=count_iteration,
    )
    return solution_gpu, "asm_cg", iteration_count[0], int(info)


def solve_case(
    mesh_number,
    order,
    mesh_root,
    direct_threshold,
    cg_rtol,
    cg_maxiter,
    verbose=True,
):
    """Run the full HDG pipeline once and return its paper-facing metrics."""
    nodes, connectivity, _, mesh_path = load_paper_mesh(
        mesh_number,
        order,
        mesh_root,
    )
    n_elements = connectivity.shape[0]
    tau = np.ones((n_elements, N_FACES_PER_ELEMENT), dtype=np.float32)
    reference_element = create_reference_element_tri(order)

    # The paper timing excludes file input, reference-element construction,
    # JIT compilation, error integration, and residual diagnostics.  It includes
    # topology preprocessing, assembly/condensation, the trace solve, elemental
    # recovery, and local postprocessing.
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()

    faces, face_info, face_neighbors_gpu = hdg_preprocess(connectivity)
    exterior_faces_host = as_numpy(face_info["extFaces"]).astype(
        np.int32, copy=False
    )

    K_gpu, Ke, f_gpu, fe, QQ_gpu, UU_gpu, Qf_gpu, Uf_gpu = matrix_assembly(
        nodes,
        connectivity,
        faces,
        reference_element,
        face_info,
        N_COMPONENTS,
        MU,
        tau,
    )
    u_dirichlet_gpu = compute_projection_faces_gpu(
        exterior_faces_host,
        nodes,
        connectivity,
        reference_element,
    )

    n_nodes_per_face = order + 1
    n_interior_faces = int(face_info["intFaces"].shape[0])
    n_exterior_faces = int(exterior_faces_host.shape[0])
    n_dof_interior = n_interior_faces * n_nodes_per_face
    n_dof_exterior = n_exterior_faces * n_nodes_per_face
    n_dof_total = n_dof_interior + n_dof_exterior

    unknown_dofs_gpu = cp.arange(n_dof_interior, dtype=INT_DTYPE)
    dirichlet_dofs_gpu = cp.arange(
        n_dof_interior, n_dof_total, dtype=INT_DTYPE
    )
    K_uu_gpu = K_gpu[unknown_dofs_gpu[:, None], unknown_dofs_gpu].tocsr()
    K_ud_gpu = K_gpu[unknown_dofs_gpu[:, None], dirichlet_dofs_gpu]
    f_reduced_gpu = f_gpu[unknown_dofs_gpu] - K_ud_gpu.dot(u_dirichlet_gpu)

    faces_gpu = cp.asarray(faces, dtype=INT_DTYPE)
    interior_faces_gpu = cp.asarray(face_info["intFaces"], dtype=INT_DTYPE)
    exterior_faces_gpu = cp.asarray(exterior_faces_host, dtype=INT_DTYPE)
    lambda_unknown_gpu, solver, iterations, cg_info = solve_trace_system(
        K_uu_gpu,
        f_reduced_gpu,
        Ke,
        fe,
        faces_gpu,
        interior_faces_gpu,
        exterior_faces_gpu,
        face_neighbors_gpu,
        n_elements,
        n_nodes_per_face,
        n_interior_faces,
        n_dof_total,
        direct_threshold,
        cg_rtol,
        cg_maxiter,
    )

    lambda_full_gpu = cp.empty(n_dof_total, dtype=FLOAT_DTYPE)
    lambda_full_gpu[:n_dof_interior] = lambda_unknown_gpu
    lambda_full_gpu[n_dof_interior:] = u_dirichlet_gpu
    u_element_gpu, q_element_gpu = compute_elements_solution_gpu(
        lambda_full_gpu,
        UU_gpu,
        QQ_gpu,
        Uf_gpu,
        Qf_gpu,
        faces_gpu,
    )

    reference_element_star = create_reference_element_tri_star(reference_element)
    reference_element_star_gpu = {}
    for name, value in reference_element_star.items():
        if isinstance(value, np.ndarray):
            dtype = INT_DTYPE if np.issubdtype(value.dtype, np.integer) else FLOAT_DTYPE
            reference_element_star_gpu[name] = cp.asarray(value, dtype=dtype)
        else:
            reference_element_star_gpu[name] = value

    nodes_gpu = cp.asarray(nodes, dtype=FLOAT_DTYPE)
    connectivity_gpu = cp.asarray(connectivity, dtype=INT_DTYPE)
    u_element_gpu = cp.asarray(u_element_gpu, dtype=FLOAT_DTYPE).ravel()
    q_element_gpu = cp.asarray(q_element_gpu, dtype=FLOAT_DTYPE).reshape(-1, 2)
    u_star_gpu = HDG_postprocess_gpu(
        nodes_gpu,
        connectivity_gpu,
        u_element_gpu,
        -q_element_gpu,
        reference_element_star_gpu,
    )
    n_star_nodes = int(reference_element_star_gpu["N"].shape[1])
    u_star_gpu = u_star_gpu.reshape(n_elements, n_star_nodes)

    cp.cuda.Stream.null.synchronize()
    pipeline_time = time.perf_counter() - start

    # All verification quantities are evaluated after the timed interval.  The
    # diagnostic integrations use FP64 arithmetic but never change the FP32
    # matrices or solution fields.
    error = compute_L2_norm(
        reference_element,
        nodes,
        connectivity,
        u_element_gpu,
        analytical_poisson,
        dtype=cp.float64,
    )
    error_postprocessed = compute_L2_norm_postprocess(
        reference_element_star,
        nodes,
        connectivity,
        u_star_gpu.reshape(-1),
        analytical_poisson,
        dtype=cp.float64,
    )
    error_flux = compute_flux_l2_error(
        reference_element,
        nodes,
        connectivity,
        q_element_gpu,
        mu=MU,
        dtype=cp.float64,
    )
    conservation = compute_local_conservation(
        Ke,
        fe,
        lambda_full_gpu,
        faces_gpu,
        reference_element,
        nodes,
        connectivity,
        mu=MU,
        dtype=cp.float64,
    )

    residual32, residual64 = evaluate_true_residual(
        K_uu_gpu, lambda_unknown_gpu, f_reduced_gpu
    )
    result = {
        "order": order,
        "mesh": mesh_number,
        "mesh_file": str(mesh_path),
        "ndof_trace": n_dof_total,
        "solver": solver,
        "cg_rtol_requested": cg_rtol if solver == "asm_cg" else "",
        "cg_info": cg_info,
        "iterations": iterations,
        "time_pipeline": pipeline_time,
        "error_u": float(error),
        "error_q": float(error_flux),
        "error_post": float(error_postprocessed),
        "residual_fp32": residual32,
        "residual_fp64_evaluated": residual64,
        **conservation,
    }
    if verbose:
        print(
            f"p={order} mesh={mesh_number} ndof={n_dof_total} "
            f"solver={solver} iterations={iterations} info={cg_info} "
            f"time_pipeline={pipeline_time:.6e} error_u={error:.6e} "
            f"error_q={error_flux:.6e} error_post={error_postprocessed:.6e} "
            f"true_residual={residual64:.6e} "
            f"conservation_l2={conservation['conservation_relative_l2']:.6e} "
            f"conservation_linf={conservation['conservation_relative_linf']:.6e}"
        )
    return result


def write_rows(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_runs(run_rows, repeats):
    summaries = []
    keys = sorted({(row["order"], row["mesh"]) for row in run_rows})
    for order, mesh in keys:
        group = [
            row for row in run_rows
            if row["order"] == order and row["mesh"] == mesh
        ]
        times = np.array([row["time_pipeline"] for row in group])
        iterations = np.array([row["iterations"] for row in group])
        summaries.append(
            {
                "order": order,
                "mesh": mesh,
                "mesh_file": group[0]["mesh_file"],
                "ndof_trace": group[0]["ndof_trace"],
                "solver": group[0]["solver"],
                "repeats": repeats,
                "cg_rtol_requested": group[0]["cg_rtol_requested"],
                "cg_info_max": max(row["cg_info"] for row in group),
                "iterations_median": float(np.median(iterations)),
                "time_median": float(np.median(times)),
                "time_q25": float(np.percentile(times, 25)),
                "time_q75": float(np.percentile(times, 75)),
                "error_u_median": float(np.median([row["error_u"] for row in group])),
                "error_q_median": float(np.median([row["error_q"] for row in group])),
                "error_post_median": float(
                    np.median([row["error_post"] for row in group])
                ),
                "residual_fp32_median": float(
                    np.median([row["residual_fp32"] for row in group])
                ),
                "residual_fp64_evaluated_median": float(
                    np.median([row["residual_fp64_evaluated"] for row in group])
                ),
                "residual_fp64_evaluated_max": max(
                    row["residual_fp64_evaluated"] for row in group
                ),
                "conservation_relative_l2_median": float(np.median(
                    [row["conservation_relative_l2"] for row in group]
                )),
                "conservation_relative_linf_median": float(np.median(
                    [row["conservation_relative_linf"] for row in group]
                )),
                "conservation_max_abs_max": max(
                    row["conservation_max_abs"] for row in group
                ),
            }
        )
    return summaries


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meshes", nargs="+", type=int, default=list(range(1, 9)))
    parser.add_argument("--orders", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup-mesh", type=int, default=5)
    parser.add_argument("--direct-threshold", type=int, default=1800)
    parser.add_argument("--cg-rtol", type=float, default=1.0e-6)
    parser.add_argument("--cg-maxiter", type=int, default=20000)
    parser.add_argument("--mesh-root", type=Path, default=Path("mesh"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/paper_unified_fp32_summary.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least one")

    print("Compilation warm-up: one full ASM-CG run for every polynomial degree")
    for order in args.orders:
        solve_case(
            args.warmup_mesh,
            order,
            args.mesh_root,
            args.direct_threshold,
            args.cg_rtol,
            args.cg_maxiter,
            verbose=False,
        )
        print(f"  warmed p={order} on mesh {args.warmup_mesh}")

    run_rows = []
    for order in args.orders:
        for mesh in args.meshes:
            for repeat in range(1, args.repeats + 1):
                row = solve_case(
                    mesh,
                    order,
                    args.mesh_root,
                    args.direct_threshold,
                    args.cg_rtol,
                    args.cg_maxiter,
                )
                row["repeat"] = repeat
                run_rows.append(row)

    run_fields = [
        "order", "mesh", "repeat", "mesh_file", "ndof_trace", "solver",
        "cg_rtol_requested", "cg_info", "iterations", "time_pipeline", "error_u",
        "error_q", "error_post", "residual_fp32", "residual_fp64_evaluated",
        "conservation_relative_l2", "conservation_relative_linf",
        "conservation_max_abs", "conservation_source_l2",
    ]
    runs_path = args.output.with_name(f"{args.output.stem}_runs.csv")
    write_rows(runs_path, run_rows, run_fields)

    summaries = summarize_runs(run_rows, args.repeats)
    summary_fields = list(summaries[0].keys())
    write_rows(args.output, summaries, summary_fields)
    print(f"Wrote per-run data to {runs_path}")
    print(f"Wrote summary data to {args.output}")


if __name__ == "__main__":
    main()
