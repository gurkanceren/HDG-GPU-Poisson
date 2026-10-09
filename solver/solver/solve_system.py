import time
import cupy as cp
# from cupyx.scipy.sparse.linalg import cg, LinearOperator, spsolve
from cupyx.scipy.sparse.linalg import LinearOperator, spsolve
import cupyx.scipy.sparse.linalg as cpsa
import cupyx.scipy.sparse as cpx_sparse
from cupyx.scipy.sparse.linalg import spsolve

from .preconditioners.block_jacobi import create_block_jacobi_preconditioner
from .preconditioners.point_jacobi import create_point_jacobi_preconditioner
from .dense_blocks.linear_operator import create_linear_operator_K
from .tracker import CGTracker
from .dense_blocks.matvec import linearOperator_RawKernel
from .cg_new import cg
# from .cg_new_timer import cg



def solve_reduced_system(
                         K_uu_gpu, f_red_gpu,
                         F_neighbors,
                         ASM_red,
                         ndof_uhat, nIntFaces, nNodesPerFace,
                         precond
                         ):
    """
    Solves the linear HDG system with different methods depending on the number of DoFs to maximize performance

    Note: As of 2026/02/26, the solution of the direct solver is better up to and including mesh4_P1 (ndof_uhat = 1600)
    """

    # Number of DoFs where optimal solver changes
    # To only use one solver, choose 0 on the solver to omit it or use a large number (2**30) to select it
    dofs_direct = 1800  # I think this is the break point where the optimal solution changes
    #dofs_direct = 1800
    # Choose the solver
    if ndof_uhat <= dofs_direct:
    # if 1 == 1:    
        u_lambda_gpu = solve_with_direct(K_uu_gpu, f_red_gpu)
    else:
        u_lambda_gpu = solve_with_cg(
                            K_uu_gpu, f_red_gpu,
                            F_neighbors,
                            ASM_red,
                            nIntFaces, nNodesPerFace,
                            precond=precond
                            )

    return u_lambda_gpu




def solve_with_cg(
        K_uu_gpu, f_red_gpu,
        F_neighbors,
        ASM_red,
        nIntFaces, nNodesPerFace,
        precond
        ):
    """
    Solves the global HDG system K_uu * u_lambda = f_red using the conjugate gradient method.
    """


    tic_0 = time.perf_counter()

    tracker = CGTracker()

    nNeighbors = 5  # Test

    # Create Linear Operator for the K matrix following X.Roca's paper
    # K_LinOp = create_linear_operator_K(K_uu_gpu, F_neighbors, nIntFaces, nNodesPerFace, tracker)
    K_dbf = linearOperator_RawKernel(K_uu_gpu, F_neighbors, nIntFaces, nNodesPerFace, nNeighbors, tracker)


    tic_1 = time.perf_counter()
    # Create a preconditioner to aid in the solution of the system
    # Preconditioner can be included in the LinearOperator form for improved performance
    if precond == "None":
        M = None
    elif precond == "pointJacobi":
        M = create_point_jacobi_preconditioner(K_uu_gpu, nIntFaces, nNodesPerFace)
    elif precond == "blockJacobi":
        M = create_block_jacobi_preconditioner(K_uu_gpu, nIntFaces, nNodesPerFace)
    elif precond == "ASM":
        M = ASM_red
    else:
        raise ValueError("This preconditioner is not available. Try:\n -None\n -pointJacobi\n -blockJacobi\n -ASM\n")

    tic_2 = time.perf_counter()

    # CG: K_uu is positive symmetric when tau>0
    # u_lambda_gpu, info = cg(K_uu_gpu, f_red_gpu, M=M, atol=1e-6, maxiter=2000, callback=tracker)
    u_lambda_gpu, info = cg(K_dbf, f_red_gpu, M=M, atol=1e-6, maxiter=2000, callback=tracker)
    # u_lambda_gpu, info = cg(K_LinOp, f_red_gpu, M=M, tol=1e-7, maxiter=2000, callback=tracker)
        
    if info != 0:
        print(f"[CG] terminó con info={info} (0=OK, >0=iter máx alcanzadas)")

    tic_3 = time.perf_counter()

    # Print the times of each function
    total_time = tic_3 - tic_0
    percent = lambda t: t / total_time * 100

    # print(f"\nEvaluating the Linear System Solver (CG with {precond} preconditioner)")
    # print(f"\t \t Conjugate Gradient converged in {tracker.iterations} iterations. \n")
    # print(f"\t \t Part 1: Creation of K Linear Operator")
    # print(f"\t \t Taken up {percent(tic_1 - tic_0):.1f}% ({tic_1 - tic_0:.3e}s)")
    # print(f"\t \t Part 2: Creation of Pre-conditioner")
    # print(f"\t \t Taken up {percent(tic_2 - tic_1):.1f}% ({tic_2 - tic_1:.3e}s)")
    # print(f"\t \t Part 3: Solving the Conjugate Gradient")
    # print(f"\t \t Taken up {percent(tic_3 - tic_2):.1f}% ({tic_3 - tic_2:.3e}s)")
    # print(f"\n \t \t Function took {total_time:.4f}s\n \n")

    # tracker.relative_times()

    return u_lambda_gpu




### DIRECT SOLVER IS VERY VERY SLOW FOR LARGE MESHSES (X50 TIMES SLOWER FOR MESH6_P4) ###
def solve_with_direct(K_uu_gpu, f_red_gpu):
    """
    Solves the global HDG system K_uu * u_lambda = f_red using a direct sparse solver.
    """

    tic_0 = time.perf_counter()

    # cuSOLVER expects Compressed Sparse Row (CSR) format for maximum efficiency.
    if not cpx_sparse.isspmatrix_csr(K_uu_gpu):
        K_uu_gpu = K_uu_gpu.tocsr()

    tic_1 = time.perf_counter()

    # Solving via cuSOLVER
    u_lambda_gpu = spsolve(K_uu_gpu, f_red_gpu)
    
    ##test start ##
    K64 = K_uu_gpu.astype(cp.float64)
    x64 = u_lambda_gpu.astype(cp.float64)
    b64 = f_red_gpu.astype(cp.float64)

    residual64 = K64.dot(x64) - b64

    relative_residual64 = (
    cp.linalg.norm(residual64)
    / cp.linalg.norm(b64)
    )

    print(
    "FP64-evaluated relative residual:",
    float(relative_residual64.get()),
    )
    # test end #
    
    
    tic_2 = time.perf_counter()

    # Print the times of each section for benchmarking
    total_time = tic_2 - tic_0
    percent = lambda t: (t / total_time * 100) if total_time > 0 else 0.0

    # print(f"\nEvaluating the Linear System Solver (Direct Sparse Solver)")
    # print(f"\t \t Part 1: Matrix Format Verification (CSR)")
    # print(f"\t \t Taken up {percent(tic_1 - tic_0):.1f}% ({tic_1 - tic_0:.3e}s)")
    # print(f"\t \t Part 2: cuSOLVER Factorization & Substitution")
    # print(f"\t \t Taken up {percent(tic_2 - tic_1):.1f}% ({tic_2 - tic_1:.3e}s)")
    # print(f"\n \t \t Function took {total_time:.4f}s\n \n")

    return u_lambda_gpu
   
