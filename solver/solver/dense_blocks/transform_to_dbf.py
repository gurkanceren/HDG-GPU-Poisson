import cupy as cp
from numba import cuda
import math




def matrix_to_dbf(K_csr, face_connectivity, nIntFaces, nNodesPerFace, nNeighbors=5):
    """
    Allocates the DBF array and runs the extraction kernel.
    """
    m = nNodesPerFace
    r = nIntFaces
    q = nNodesPerFace * nNeighbors

    # Allocate in Fortran order for column-major coalescing
    # Initialize with zeros to automatically handle any sparsity within the dense blocks
    K_dbf = cp.zeros((m, r, q), dtype=K_csr.dtype, order='F')

    # Configure launch dimensions (1 thread per matrix row)
    threads_per_block = 256
    total_rows = nIntFaces * nNodesPerFace
    blocks_per_grid = math.ceil(total_rows / threads_per_block)

    # Launch
    transform_csr_to_dbf_kernel[blocks_per_grid, threads_per_block](
        K_csr.indptr, K_csr.indices, K_csr.data, 
        K_dbf, face_connectivity, 
        nNodesPerFace, nIntFaces, nNeighbors
    )
    
    cuda.synchronize()
    return K_dbf


def vector_to_dbf(x, X_inflated, face_connectivity, m, r, q, nNodesPerFace, nNeighbors=5):
    """
    Allocates the inflated vector array and runs the inflation kernel.
    """
    # Fortran ordered array (assume it as an input)
    # It would be too expensive to allocate memory on each iteration
    # X_inflated = cp.empty((q, r), dtype=x.dtype, order='F')
    
    # Configure launch dimensions (1 thread per face)
    threads_per_block = 256
    blocks_per_grid = math.ceil(r / threads_per_block)
    
    # Launch
    transform_vector_to_dbf_kernel[blocks_per_grid, threads_per_block](
        x, X_inflated, face_connectivity, nNodesPerFace, r, nNeighbors
    )
    
    # cuda.synchronize()
    # return X_inflated



############################################################
# ------------------ INDIVIDUAL KERNELS ------------------ #
############################################################

@cuda.jit
def transform_csr_to_dbf_kernel(indptr, indices, data, K_dbf, F_neighbors, nNodesPerFace, nIntFaces, nNeighbors):
    """
    Maps non-zero entries from a CSR matrix into the Dense Block Format (DBF) multi-array.

    The stiffness matrix contraction and solution vector inflation come from the paper:
    X. Roca, N. C. Nguyen and J. Peraire. GPU-accelerated sparse matrix-vector product for a
    hybridizable discontinuous Galerkin method.

    Link to pdf here: https://www.mit.edu/~cuongng/project/sc3/sc3.pdf
    """
    global_row = cuda.grid(1)

    if global_row < nIntFaces * nNodesPerFace:
        # Determine which face and local row this thread belongs to
        face_idx = global_row // nNodesPerFace
        local_row = global_row % nNodesPerFace

        row_start = indptr[global_row]
        row_end = indptr[global_row + 1]

        # Scan the CSR columns for this row
        for idx in range(row_start, row_end):
            global_col = indices[idx]
            col_face = global_col // nNodesPerFace
            local_col = global_col % nNodesPerFace

            # Determine which neighbor block (k_a) this column belongs to
            neighbor_idx = -1
            for k in range(nNeighbors):
                if F_neighbors[face_idx, k] == col_face:
                    neighbor_idx = k
                    break
            
            # If the column belongs to a valid neighbor, insert it into the DBF array
            if neighbor_idx != -1:
                # Calculate the exact column index (q) in the DBF array
                q_idx = local_col + nNodesPerFace * neighbor_idx
                
                # Write to DBF tensor: K_dbf[m, r, q]
                K_dbf[local_row, face_idx, q_idx] = data[idx]


@cuda.jit
def transform_vector_to_dbf_kernel(x, X_inflated, F_neighbors, nNodesPerFace, nIntFaces, nNeighbors):
    """
    Inflates a 1D vector 'x' into the redundant q x r Dense Block Format vector 'X_inflated'.

    The stiffness matrix contraction and solution vector inflation come from the paper:
    X. Roca, N. C. Nguyen and J. Peraire. GPU-accelerated sparse matrix-vector product for a
    hybridizable discontinuous Galerkin method.

    Link to pdf here: https://www.mit.edu/~cuongng/project/sc3/sc3.pdf
    """
    face_idx = cuda.grid(1)
    
    if face_idx < nIntFaces:
        for k_a in range(nNeighbors):
            neighbor_face = F_neighbors[face_idx, k_a]
            
            if neighbor_face != -1:
                # Copy the neighbor's degrees of freedom into this face's block
                for k_p in range(nNodesPerFace):
                    q_idx = k_p + k_a * nNodesPerFace
                    global_x_idx = neighbor_face * nNodesPerFace + k_p
                    X_inflated[q_idx, face_idx] = x[global_x_idx]
            else:
                # Pad with zeros for boundary faces (non-existent neighbors)
                for k_p in range(nNodesPerFace):
                    q_idx = k_p + k_a * nNodesPerFace
                    X_inflated[q_idx, face_idx] = 0.0
