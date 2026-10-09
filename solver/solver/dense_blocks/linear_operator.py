from math import ceil
import time

import cupy as cp
from numba import cuda
from cupyx.scipy.sparse.linalg import LinearOperator

from .transform_to_dbf import matrix_to_dbf, vector_to_dbf
from ..tracker import CGTracker


def create_linear_operator_K(K_csr, F_neighbors, nIntFaces, nNodesPerFace, tracker):
    nDofsInt = nIntFaces * nNodesPerFace
    m = nNodesPerFace
    r = nIntFaces
    nNeighbors = 5  
    q = nNodesPerFace * nNeighbors

    K_dbf = matrix_to_dbf(K_csr, F_neighbors, nIntFaces, nNodesPerFace, nNeighbors=5)

    # Only allocate the output array
    Y = cp.empty((m, r), dtype=K_dbf.dtype, order='F')

    threads_per_block = 256
    total_threads = m * r
    blocks_per_grid = ceil(total_threads / threads_per_block)

    def _matrix_vector_op(x):
        tic = time.perf_counter()
        
        # Execute the fused operation
        fused_dbf_matvec_kernel[blocks_per_grid, threads_per_block](
            K_dbf, x, Y, F_neighbors, m, r, nNodesPerFace, nNeighbors
        )
        
        tracker.matvec += time.perf_counter() - tic

        tic = time.perf_counter()
        Y_ravel = Y.ravel(order='F')
        tracker.ravel += time.perf_counter() - tic
        
        return Y_ravel

    K_LinOp = LinearOperator(shape=(nDofsInt, nDofsInt), matvec=_matrix_vector_op, dtype=K_dbf.dtype)

    return K_LinOp


@cuda.jit
def fused_dbf_matvec_kernel(K_dbf, x, Y, F_neighbors, m, r, nNodesPerFace, nNeighbors):
    """
    Fused Matrix-Vector DBF multiplication. 
    Reads the 1D vector 'x' on the fly using face connectivity to bypass the inflation step entirely.
    """
    # Global thread ID calculation
    t = cuda.grid(1)
    
    # Map thread to output matrix coordinates 
    i = t % m
    l = t // m
    
    if l < r:
        y_val = 0.0
        
        # Loop over the neighbor blocks
        for k_a in range(nNeighbors):
            neighbor_face = F_neighbors[l, k_a]
            
            if neighbor_face != -1:
                # Loop through the nodes of this specific neighbor
                for k_p in range(nNodesPerFace):
                    # q_idx is the column coordinate in the DBF matrix K
                    q_idx = k_p + k_a * nNodesPerFace
                    
                    # global_x_idx is the exact coordinate in the 1D residual vector x
                    global_x_idx = neighbor_face * nNodesPerFace + k_p
                    
                    # Read directly from x and multiply by the DBF matrix entry
                    y_val += K_dbf[i, l, q_idx] * x[global_x_idx]
                    
        # Write to the Fortran-ordered global output array (Perfectly coalesced)
        Y[i, l] = y_val

