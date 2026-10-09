from math import ceil
import time

import cupy as cp
from numba import cuda
from cupyx.scipy.sparse.linalg import LinearOperator

from .transform_to_dbf import matrix_to_dbf, vector_to_dbf
from ..tracker import CGTracker


def create_linear_operator_K(K_csr, F_neighbors, nIntFaces, nNodesPerFace, tracker):
    """
    This function creates a sparse matrix-vector product Linear Operator to improve
    convergence of the iterative solvers.

    The stiffness matrix contraction and solution vector inflation come from the paper:
    X. Roca, N. C. Nguyen and J. Peraire. GPU-accelerated sparse matrix-vector product for a
    hybridizable discontinuous Galerkin method.

    Link to pdf here: https://www.mit.edu/~cuongng/project/sc3/sc3.pdf
    """

    nDofsInt = nIntFaces * nNodesPerFace

    # Define the contraction dimensions
    m = nNodesPerFace
    r = nIntFaces
    nNeighbors = 5  # For a 2D triangle, each face has 5 shared faces (counting itself)
    q = nNodesPerFace * nNeighbors

    # Contract the matrix K to be compatible with the operation
    K_dbf = matrix_to_dbf(K_csr, F_neighbors, nIntFaces, nNodesPerFace, nNeighbors=5)


    # Pre-allocate the output array Y (Column-mayor ordering, as said in the paper)
    Y = cp.empty((m, r), dtype=K_dbf.dtype, order='F')
    X_inflated = cp.empty((q, r), dtype=K_dbf.dtype, order='F')

    # Block and grid dimensions
    threads_per_block = 256
    total_threads = m * r
    blocks_per_grid = ceil(total_threads / threads_per_block)

    # Create the matrix-vector operator
    def _matrix_vector_op(x):
        tic = time.perf_counter()
        # Inflate the vector x to be compatible with the matrix K
        # X_inflated = vector_to_dbf(x, F_neighbors, m, r, q, nNodesPerFace, nNeighbors=5)
        vector_to_dbf(x, X_inflated, F_neighbors, m, r, q, nNodesPerFace, nNeighbors=5)
        tracker.vector_to_dbf += time.perf_counter() - tic

        tic = time.perf_counter()
        # Perform the solution of the contracted matrix and inflated vector
        matrix_vector_operation_kernel[blocks_per_grid, threads_per_block](K_dbf, X_inflated, Y, m, r, q)
        # cuda.synchronize()
        tracker.matvec += time.perf_counter() - tic

        # Transform back into a 1D vector for the CG to do next step
        tic = time.perf_counter()
        Y_ravel = Y.ravel(order='F')
        tracker.ravel += time.perf_counter() - tic
        return Y_ravel

    # Create the linear operator
    K_LinOp = LinearOperator(shape=(nDofsInt, nDofsInt), matvec=_matrix_vector_op, dtype=K_dbf.dtype)

    return K_LinOp




@cuda.jit
def matrix_vector_operation_kernel(K, X, Y, m, r, q):
    """
    Matrix-vector multiplication for block dense format (GPU)
   
    The stiffness matrix contraction and solution vector inflation come from the paper:
    X. Roca, N. C. Nguyen and J. Peraire. GPU-accelerated sparse matrix-vector product for a
    hybridizable discontinuous Galerkin method.

    Link to pdf here: https://www.mit.edu/~cuongng/project/sc3/sc3.pdf
    
    Data:
        K: System matrix in DBF, shape (m, r, q)
        X: Inflated input vector in DBF, shape (q, r)
        Y: Output vector, shape (m, r)
        m: Number of rows per face block (nNodesPerFace * nComponents)
        r: Number of faces (nIntFaces)
        q: Total columns per block row (nNodesPerFace * nComponents * nNeighbors)
    """

    # Global thread ID calculation
    t = cuda.grid(1)
    
    # Map thread to output matrix coordinates 
    i = t % m
    l = t // m
    
    # Guard against out-of-bounds threads
    if l < r:
        # Initialize accumulator
        y_val = 0.0
        
        # Loop over the dense block columns
        for k in range(q):
            # Read and accumulate 
            y_val += K[i, l, k] * X[k, l]
            
        # Write to the global output array
        Y[i, l] = y_val
