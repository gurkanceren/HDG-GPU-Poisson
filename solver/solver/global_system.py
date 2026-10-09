import time

import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from numba import cuda



def assemble_K_csr_disordered(Ke,
                   Faces,
                   intFaces, extFaces,
                   nNodesPerFace, nFacesPerEl,
                   intDtype, floatDtype
                   ):
    """
    This function transforms the elemental stifness matrix into a CSR format
    """

    # ----------------------------- #
    #   Obtain data from matrices   #
    # ----------------------------- #
    tic_0 = time.perf_counter()

    # Number of degrees of freedom
    nIntFaces = intFaces.shape[0]
    nExtFaces = extFaces.shape[0]
    nTotalFaces = nIntFaces + nExtFaces

    nDofInt = nIntFaces * nNodesPerFace
    nDofExt = nExtFaces * nNodesPerFace
    nDofTotal = nDofInt + nDofExt

    # Number of neighbor Dof of each Dof
    nNeighbors = 2*nFacesPerEl 
    nNeighborInt = nNeighbors * nNodesPerFace
    nNeighborExt = nFacesPerEl * nNodesPerFace

    nIntEntries = nNeighborInt * nIntFaces * nNodesPerFace
    nExtEntries = nNeighborExt * nExtFaces * nNodesPerFace  
    nCsrEntries = nIntEntries + nExtEntries


    # ------------------------- #
    #   Initialize CSR matrix   #
    # ------------------------- #
    tic_1 = time.perf_counter()

    # Initialize the indptr, cols, and values vectors
    K_indptr = cp.empty(nDofTotal + 1, dtype=intDtype)
    K_cols   = cp.empty(nCsrEntries,   dtype=intDtype)
    K_vals   = cp.empty(nCsrEntries,   dtype=floatDtype)

    # -------------------------- #
    #   Create K_indptr vector   #
    # -------------------------- #
    tic_2 = time.perf_counter()

    # Indptr is just a linspace of the interior + exterior entries
    # NOTE: Trying to get a CSR with repeated and disordered values
    K_indptr[:(nDofInt + 1)] = cp.arange(0, nDofInt + 1) * nNeighborInt
    K_indptr[(nDofInt + 1):] = cp.arange(1, nDofExt + 1) * nNeighborExt + nDofInt*nNeighborInt

    # -------------------------------------- #
    #   Introduce values of internal faces   #
    # -------------------------------------- #
    tic_3 = time.perf_counter()

    # Create the values and columns with a kernel in parallel
    # 3D block to improve performance (2 elements sharing a face)
    threads_per_block = (nNodesPerFace, nFacesPerEl, 2) 
    blocks_per_grid = nIntFaces
    isIntFace = True

    K_csr_kernel[blocks_per_grid, threads_per_block](
        Ke, Faces, K_indptr,
        intFaces, extFaces,
        nNodesPerFace, nIntFaces, nFacesPerEl,
        K_cols, K_vals,             # Output
        isIntFace                   # Flag to have use this kernel for internal and external faces
        )

    # -------------------------------------- #
    #   Introduce values of external faces   #
    # -------------------------------------- #
    tic_4 = time.perf_counter()

    # Create the values and columns with a kernel in parallel
    threads_per_block = (nNodesPerFace, nFacesPerEl, 1) # 3D block to improve performance (1 elements sharing a boundary face)
    blocks_per_grid = nExtFaces
    isIntFace = False

    K_csr_kernel[blocks_per_grid, threads_per_block](
        Ke, Faces, K_indptr,
        intFaces, extFaces,
        nNodesPerFace, nIntFaces, nFacesPerEl,
        K_cols, K_vals,             # Output
        isIntFace                   # Flag to have use this kernel for internal and external faces
        )

    # --------------------------- #
    #   Assemble the CSR matrix   #
    # --------------------------- #
    tic_5 = time.perf_counter()

    # Create the CSR matrix
    K_csr = csr_matrix((K_vals, K_cols, K_indptr))
    # print(f"\n\nThis is the K_indptr (New): \n{K_indptr}\n")
    # print(f"This is the K_cols (Non-duplicates): \n{K_cols}\n")
    # print(f"This is the K_vals (Non-duplicates): \n{K_vals}\n")

    # ---------------------------------- #
    #   Make the CSR matrix cannonical   #
    # ---------------------------------- #
    tic_6 = time.perf_counter()

    K_csr.sum_duplicates()
    tic_7 = time.perf_counter()
    K_csr.sort_indices()
    # print(f"This is the K_indptr (New 2): \n{K_csr.indptr}\n\n")
    # print(f"This is the K_cols (New 2): \n{K_csr.indices}\n\n")
    # print(f"This is the K_vals (New 2): \n{K_csr.data}\n\n")

    # ------------------------ #
    #   Timming the function   #
    # ------------------------ #

    # Print the times of each function
    tic_8 = time.perf_counter()
    total_time = tic_8 - tic_0
    percent = lambda t: t / total_time * 100

    # print(f"\n \t Evaluating the CSR matrix assembly")
    # print(f"\t \t \t Part 1: Obtain data from matrices")
    # print(f"\t \t \t Taken up {percent(tic_1 - tic_0):.1f}% ({tic_1 - tic_0:.3e}s)")
    # print(f"\t \t \t Part 2: Initialize CSR matrix")
    # print(f"\t \t \t Taken up {percent(tic_2 - tic_1):.1f}% ({tic_2 - tic_1:.3e}s)")
    # print(f"\t \t \t Part 3: Create K_indptr vector")
    # print(f"\t \t \t Taken up {percent(tic_3 - tic_2):.1f}% ({tic_3 - tic_2:.3e}s)")
    # print(f"\t \t \t Part 4: Introduce values of internal faces")
    # print(f"\t \t \t Taken up {percent(tic_4 - tic_3):.1f}% ({tic_4 - tic_3:.3e}s)")
    # print(f"\t \t \t Part 5: Introduce values of external faces")
    # print(f"\t \t \t Taken up {percent(tic_5 - tic_4):.1f}% ({tic_5 - tic_4:.3e}s)")
    # print(f"\t \t \t Part 6: Assemble CSR matrix")
    # print(f"\t \t \t Taken up {percent(tic_6 - tic_5):.1f}% ({tic_6 - tic_5:.3e}s)")
    # print(f"\t \t \t Part 7: Make CSR matrix cannonical (Sum duplicates)")
    # print(f"\t \t \t Taken up {percent(tic_7 - tic_6):.1f}% ({tic_7 - tic_6:.3e}s)")
    # print(f"\t \t \t Part 7: Make CSR matrix cannonical (Sort indices)")
    # print(f"\t \t \t Taken up {percent(tic_8 - tic_7):.1f}% ({tic_8 - tic_7:.3e}s)")
    # print(f"\t \n \t \t Function took {total_time:.4f}s\n \n")


    return K_csr









@cuda.jit
def K_csr_kernel(
        Ke, Faces, K_indptr,
        intFaces, extFaces,
        nNodesPerFace, nIntFaces, nFacesPerEl,
        K_cols, K_vals,             # Output
        isIntFace=True              # Flag to have use this kernel for internal and external faces
        ):
    """
    This kernel creates the values and columns vector for the K_csr matrix in parallel.

    Note: Need to be run twice, one for intFaces and one for extFaces
    """

    # Thread identification
    intExtFace = cuda.blockIdx.x                # Face index of the internal or external faces
    localNeighborNode = cuda.threadIdx.x        # Local node of the neighboring elements
    localNeighborFace = cuda.threadIdx.y        # Local face of the neighboring face
    localEl = cuda.threadIdx.z                  # Either 0 or 1 for the elements sharing a face

    # Obtain the column of the node in the elemental matrix
    localNodeColumn = localNeighborNode + localNeighborFace*nNodesPerFace

    # Obtain element and face data
    if isIntFace:
        globalEl = intFaces[intExtFace, 2*localEl]     # Two elements sharing a face
        localFace = intFaces[intExtFace, 2*localEl + 1]
        globalFace = intExtFace 
    else:
        globalEl = extFaces[intExtFace, 0]
        localFace = extFaces[intExtFace, 1]
        globalFace = intExtFace + nIntFaces

    globalNeighborFace = Faces[globalEl, localNeighborFace]
    globalNeighborNode = localNeighborNode + globalNeighborFace*nNodesPerFace   # Global column of this Dof

    # For loop with the number of face nodes
    for faceNode in range(nNodesPerFace):
        # Get the data from the stiffness matrix
        localNodeRow = faceNode + localFace*nNodesPerFace
        K_val = Ke[globalEl, localNodeRow, localNodeColumn]

        # Obtaian the row to write the data in the CSR vectors
        globalNode = faceNode + globalFace*nNodesPerFace 
        csrOffset = K_indptr[globalNode]
        csrRow = localNodeColumn + localEl*nNodesPerFace*nFacesPerEl + csrOffset

        # Write the values into the vectors
        K_cols[csrRow] = globalNeighborNode
        K_vals[csrRow] = K_val



