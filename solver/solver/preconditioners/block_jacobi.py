from math import ceil
import cupy as cp
import cupyx.scipy.sparse as cpsp
from cupyx.scipy.sparse.linalg import LinearOperator
from numba import cuda



def create_block_jacobi_preconditioner(K_gpu, nIntFaces, nNodesPerFace):
    """
    Creates a Block Jacobi Preconditioner for the matrix K. 
    """
    # --- Extract the diagonal blocks from the csr matrix --- #
    # Blocks and threads
    threadsPerBlock = 128
    nBlocks = ceil((nNodesPerFace * nIntFaces) / threadsPerBlock)

    # Initialize the dense blocks matrix
    blockFactor = 1
    M_block = cp.zeros((nIntFaces // blockFactor, blockFactor * nNodesPerFace, blockFactor * nNodesPerFace), dtype=K_gpu.dtype)

    extract_jacobi_blocks_multiple_thread[nBlocks, threadsPerBlock](
            K_gpu.indptr,
            K_gpu.indices,
            K_gpu.data,
            M_block,
            nIntFaces // blockFactor,
            blockFactor * nNodesPerFace)
            
    cuda.synchronize()

    # --- Invert the dense blocks independently --- #
    M_inv = cp.linalg.inv(M_block)


    # --- Change the matrix into CSR format for faster matrix-vector operations --- #
    M_BJPrec = assemble_csr_preconditioner(M_inv)
    return M_BJPrec


    # --- Create a Linear Operator to improve the performance of the CG --- #
    # M_BJ_LinOp = create_linear_operator_block_jacobi(M_inv)
    # return M_BJ_LinOp



@cuda.jit
def extract_jacobi_blocks_multiple_thread(indptr, indices, data, M_block, nIntFaces, nNodesPerFace):
    """
    Extracts dense diagonal blocks from the CSR K matrix.
    Each thread is responsible for extracting exactly one row of a dense block.
    """
    # Thread initialization
    threadIdx = cuda.grid(1)
    
    #  Guard Clause: Prevent out-of-bounds memory access before reading indptr
    if threadIdx >= nIntFaces * nNodesPerFace:
        return

    faceIdx = threadIdx // nNodesPerFace
    rowIdx = threadIdx % nNodesPerFace

    # Obtain the global indices for the columns of the CSR matrix
    globalColStart = indptr[threadIdx]    
    globalColEnd = indptr[threadIdx + 1]

    # Loop over the columns to include the diagonal blocks in the new matrix
    for idx in range(globalColStart, globalColEnd):
        globalColumn = indices[idx]
        isInBlockDiagonal = (globalColumn >= faceIdx * nNodesPerFace) and (globalColumn < (faceIdx + 1) * nNodesPerFace)

        if isInBlockDiagonal:
            localColumn = globalColumn - (faceIdx * nNodesPerFace)
            M_block[faceIdx, rowIdx, localColumn] = data[idx]


def assemble_csr_preconditioner(M_invBlocks):
    """
    Converts a 3D tensor of dense diagonal blocks into a global CSR sparse matrix.
    """

    # dtype = M_invBlocks.dtype
    dtype = cp.int32
    nIntFaces, nNodesPerFace, _ = M_invBlocks.shape
    nDofsInt = nIntFaces * nNodesPerFace
    
    # Flatten the dense blocks into the 1D data array
    data = M_invBlocks.ravel()
    
    # Each of the rows has exactly nNodesPerFace non-zero entries
    indptr = cp.arange(0, nDofsInt * nNodesPerFace + 1, nNodesPerFace, dtype=dtype)
    
    # Base columns for a single block: [0, 1, 2, ..., nNodesPerFace-1]
    base_cols = cp.arange(nNodesPerFace, dtype=dtype)
    
    # Determine the starting column offset for the block belonging to each row
    row_blocks = cp.arange(nDofsInt, dtype=dtype) // nNodesPerFace
    block_offsets = row_blocks * nNodesPerFace
    
    # Add the base columns to the block offsets and flatten
    indices = (block_offsets[:, None] + base_cols[None, :]).ravel()
    
    # 4. Construct the CSR matrix
    M_invCsr = cpsp.csr_matrix((data, indices, indptr), shape=(nDofsInt, nDofsInt))
    
    return M_invCsr



def create_linear_operator_block_jacobi(M_inv_blocks):
    """
    Wraps the 3D tensor of inverted diagonal blocks in a LinearOperator.
    This bypasses CSR memory overhead by using batched dense matrix-vector multiplication.
    """
    nIntFaces, nNodesPerFace, _ = M_inv_blocks.shape
    nDofsInt = nIntFaces * nNodesPerFace

    
    def _apply_M(r):
        # See memory usage
        # pool = cp.get_default_memory_pool()
        # mem_prev = pool.used_bytes()

        # Reshape the 1D residual vector into a batch of column vectors
        # Shape becomes: (nIntFaces, nNodesPerFace, 1)
        r_reshaped = r.reshape(nIntFaces, nNodesPerFace, 1)
        
        # Perform batched matrix-vector multiplication
        z_reshaped = cp.matmul(M_inv_blocks, r_reshaped)
        
        # See memory after
        # mem_after = pool.used_bytes()
        # allocated_mb = (mem_after - mem_prev) / (1024**2)
        # print(f"Allocated inside preconditioner: {allocated_mb:.4} MB")

        # Flatten the result back to a 1D vector
        return z_reshaped.ravel()
    
    # Return the LinearOperator object
    M_BJ_LinOp = LinearOperator(shape=(nDofsInt, nDofsInt), matvec=_apply_M, dtype=M_inv_blocks.dtype)

    return M_BJ_LinOp












# ----------- TESTING ----------- #
if __name__ == "__main__":
    # ---------------------------------------------------------- #
    # Block Jacobi Inversion test
    # ---------------------------------------------------------- #

    # # Create a simple matrix
    # A = cp.array([[1, 0], [0, 2]], dtype=cp.float32)
    # A_csr = cpsp.csr_matrix(A)
    # nIntFaces_A = 2
    # nNodesPerFace_A = 1
    # 
    # A_BJPrec = create_block_jacobi_preconditioner(A_csr, nIntFaces_A, nNodesPerFace_A)
    # print(f"This is the normal matrix: \n{A} \n")
    # print(f"This is the inverted matrix: \n{A_BJPrec.todense()} \n")
    #
    #
    # # Create another simple matrix
    # B = cp.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]], dtype=cp.float32)
    # B_csr = cpsp.csr_matrix(B)
    # nIntFaces_B = 2
    # nNodesPerFace_B = 2
    # 
    # B_BJPrec = create_block_jacobi_preconditioner(B_csr, nIntFaces_B, nNodesPerFace_B)
    # print(f"This is the normal matrix: \n{B} \n")
    # print(f"This is the inverted matrix: \n{B_BJPrec.todense()} \n")


    # ---------------------------------------------------------- #
    # Linear Operator Test
    # ---------------------------------------------------------- #
    # Create matrix
    matrix = cp.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
    vector = cp.array([1, 2, 3])
    # matrix = cp.arange(100000, dtype=cp.float32).reshape(10, 100, 100) 
    # vector = cp.ones(1000, dtype=cp.float32)

    print(f"This is the shape of the matrix: \n{matrix.shape} \n")

    M_LinOp = create_linear_operator_block_jacobi(matrix)
    result = M_LinOp.dot(vector)
    print(f"This is the matrix-vector multiplication solution: \n{result} \n")
