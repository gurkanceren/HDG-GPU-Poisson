import cupy as cp
from cupyx.scipy.sparse.linalg import LinearOperator



def create_point_jacobi_preconditioner(K_gpu, nIntFaces, nNodesPerFace):
    """
    Creates a Point Jacobi Preconditioner for the matrix K. 
    """
    # Precond Jacobi (M ≈ diag(K)^{-1})
    diag = K_gpu.diagonal()

    # No per 0 divisions
    inv_diag = 1.0 / cp.where(diag != 0, diag, 1.0)

    def _apply_M(x):
        return inv_diag * x

    K_PJ_LinOp = LinearOperator(shape=K_gpu.shape, matvec=_apply_M, dtype=K_gpu.dtype)

    return K_PJ_LinOp
