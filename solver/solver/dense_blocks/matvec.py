import math
import time

import cupy as cp
from numba import cuda
from cupyx.scipy.sparse.linalg import LinearOperator

from .transform_to_dbf import matrix_to_dbf, vector_to_dbf

def linearOperator_RawKernel(
        K_csr, F_neighbors,
        nIntFaces, nNodesPerFace, nNeighbors,
        tracker
        ):
    """
    This function creates a sparse matrix-vector product Linear Operator to improve
    convergence of the iterative solvers.

    The choice of the cp.RawKernel over a numba.cuda kernel was made due to the very high overhead
    of the numba kernels which add up fast over a large number of iterations.

    The stiffness matrix contraction and solution vector inflation come from the paper:
    X. Roca, N. C. Nguyen and J. Peraire. GPU-accelerated sparse matrix-vector product for a
    hybridizable discontinuous Galerkin method.

    Link to pdf here: https://www.mit.edu/~cuongng/project/sc3/sc3.pdf
    """

    nDofsInt = nIntFaces * nNodesPerFace

    # Contract the matrix K to be compatible with the operation
    K_dbf = matrix_to_dbf(K_csr, F_neighbors, nIntFaces, nNodesPerFace, nNeighbors)


    # Pre-allocate the output vector Y 
    y = cp.empty(nNodesPerFace*nIntFaces, dtype=K_dbf.dtype)

    # Block and grid dimensions
    threads_per_block = 256
    total_threads = nNodesPerFace*nIntFaces
    blocks_per_grid = math.ceil(total_threads / threads_per_block)

    # Create the matrix-vector operator
    def _matrix_vector_op(x):
        tracker.nMatvecOps +=1

        # Run the kernel of the matrix-vector operation
        tic = time.perf_counter()

        y = cp.empty(nNodesPerFace*nIntFaces, dtype=K_dbf.dtype)
        matrix_vector_operation_RawKernel(
        (blocks_per_grid,), (threads_per_block,),
        (K_dbf, x, F_neighbors, nNodesPerFace, nIntFaces, nNeighbors, y)
        )

        tracker.matvec += time.perf_counter() - tic

        return y

    # Create the linear operator
    K_LinOp = LinearOperator(shape=(nDofsInt, nDofsInt), matvec=_matrix_vector_op, dtype=K_dbf.dtype)

    return K_LinOp



function_code = r"""
extern "C" __global__
void matvec_multiplication(const float* K, const float* x,
                           const int* F_neighbors,
                           const int nNodesPerFace, const int nIntFaces, const int nNeighbors,
                           float* y) {
    /*
    Matrix-vector multiplication for block dense format (GPU)
   
    The stiffness matrix contraction and solution vector inflation come from the paper:
    X. Roca, N. C. Nguyen and J. Peraire. GPU-accelerated sparse matrix-vector product for a
    hybridizable discontinuous Galerkin method.

    Link to pdf here: https://www.mit.edu/~cuongng/project/sc3/sc3.pdf
    
    Data:
        K: System matrix in DBF, shape (m, r, q)
        x: Inflated input vector in DBF, shape (m*r)
        m: Number of rows per face block (nNodesPerFace * nComponents)
        r: Number of faces (nIntFaces)
        q: Total columns per block row (nNodesPerFace * nComponents * nNeighbors)
        y: Output vector, shape (m*r)
    */

    //Global thread ID calculation
    int globalThread = threadIdx.x + blockDim.x * blockIdx.x;

    // Map thread to output matrix coordinates
    int localNode = globalThread % nNodesPerFace;
    int globalFace = globalThread / nNodesPerFace;

    // Guard against out-of-bounds threads
    if (globalFace < nIntFaces) {
        // Initialize accumulator
        float y_val = 0;

        // Loop over the neighbors and the local nodes of the neighbor
        // Use double loop for more efficiency over using modulo and integer division
        for (int neighborLocalFace = 0; neighborLocalFace < nNeighbors; neighborLocalFace++) {
            // Obtain the global neighbor face index (C-indexed array)
            int neighborGlobalFace = F_neighbors[neighborLocalFace + globalFace*nNeighbors];

            for (int neighborLocalNode = 0; neighborLocalNode < nNodesPerFace; neighborLocalNode++){
                int qIdx = neighborLocalNode + nNodesPerFace*neighborLocalFace;

                // Obtain the value of the corresponding neighbor face
                float x_val = 0.0;
                if (neighborGlobalFace != -1) {
                    int globalXIdx = neighborLocalNode + neighborGlobalFace*nNodesPerFace;
                    x_val = x[globalXIdx];
                }
                
                // Obtain the index of the DBF matrix K
                int KIdx = localNode + nNodesPerFace*globalFace + (nNodesPerFace*nIntFaces)*qIdx;

                // Read and accumulate
                y_val += K[KIdx] * x_val;
            }
        }

        //Write to the global output array
        int yIdx = localNode + nNodesPerFace*globalFace;
        y[yIdx] = y_val;

    }
}
"""

matrix_vector_operation_RawKernel = cp.RawKernel(function_code, "matvec_multiplication")
