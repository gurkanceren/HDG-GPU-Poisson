import time

import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from numba import cuda, int32



def assemble_K_csr(
                   Ke, fe,
                   Faces,
                   intFaces, extFaces,
                   nNodesPerFace, nFacesPerEl,
                   intDtype, floatDtype
                   ):
    """
    This function transforms the elemental stifness matrix into a CSR format
    """

    # --- Obtain data from matrices --- #
    tic_0 = time.perf_counter()

    # Number of degrees of freedom
    nIntFaces = intFaces.shape[0]
    nExtFaces = extFaces.shape[0]
    nTotalFaces = nIntFaces + nExtFaces

    nDofInt = nIntFaces * nNodesPerFace
    nDofExt = nExtFaces * nNodesPerFace
    nDofTotal = nDofInt + nDofExt

    # Number of neighbor Dof of each Dof
    nNeighbors = 2*nFacesPerEl - 1
    nNeighborInt = nNeighbors * nNodesPerFace
    nNeighborExt = nFacesPerEl * nNodesPerFace

    nIntEntries = nNeighborInt * nIntFaces * nNodesPerFace
    nExtEntries = nNeighborExt * nExtFaces * nNodesPerFace  
    nCsrEntries = nIntEntries + nExtEntries


    # --- Initialize CSR matrix --- #
    tic_1 = time.perf_counter()

    # Initialize the indptr, cols, and values vectors
    K_indptr = cp.empty(nDofTotal + 1, dtype=intDtype)
    K_cols   = cp.empty(nCsrEntries,   dtype=intDtype)
    K_vals   = cp.zeros(nCsrEntries,   dtype=floatDtype)    # Need to be zeros for the sum of duplicates

    f = cp.zeros(nDofTotal, dtype=floatDtype)


    # --- Create K_indptr vector --- #
    tic_2 = time.perf_counter()

    # Indptr is just a linspace of the interior + exterior entries
    K_indptr[:(nDofInt + 1)] = cp.arange(0, nDofInt + 1) * nNeighborInt
    K_indptr[(nDofInt + 1):] = cp.arange(1, nDofExt + 1) * nNeighborExt + nDofInt*nNeighborInt


    # --- Introduce values of internal faces --- #
    tic_3 = time.perf_counter()

    # Create the values and columns with a kernel in parallel
    # 3D block to improve performance (2 elements sharing a face)
    threads_per_block = (nNodesPerFace, nFacesPerEl, 2) 
    blocks_per_grid = nIntFaces
    isIntFace = True

    K_csr_kernel[blocks_per_grid, threads_per_block](
        Ke, fe,
        Faces, K_indptr,
        intFaces, extFaces,
        nNodesPerFace, nIntFaces, nFacesPerEl,
        K_cols, K_vals, f,          # Output
        isIntFace                   # Flag to have use this kernel for internal and external faces
        )


    # --- Introduce values of external faces --- #
    tic_4 = time.perf_counter()

    # Create the values and columns with a kernel in parallel
    threads_per_block = (nNodesPerFace, nFacesPerEl, 1) # 3D block to improve performance (1 elements sharing a boundary face)
    blocks_per_grid = nExtFaces
    isIntFace = False

    K_csr_kernel[blocks_per_grid, threads_per_block](
        Ke, fe,
        Faces, K_indptr,
        intFaces, extFaces,
        nNodesPerFace, nIntFaces, nFacesPerEl,
        K_cols, K_vals, f,          # Output
        isIntFace                   # Flag to have use this kernel for internal and external faces
        )


    # --- Assemble the CSR matrix --- #
    tic_5 = time.perf_counter()

    # Create the CSR matrix
    K_csr = csr_matrix((K_vals, K_cols, K_indptr))
    # print(f"\n\nThis is the K_indptr (New): \n{K_indptr}\n")
    # print(f"This is the K_cols (New): \n{K_cols}\n")
    # print(f"This is the K_vals (New): \n{K_vals}\n")


    # --- Make the CSR matrix cannonical --- #
    tic_6 = time.perf_counter()
    K_csr.sort_indices()

    
    tic_7 = time.perf_counter()
    # print(f"\n\nThis is the K_indptr (Ordered): \n{K_csr.indptr}\n\n")
    # print(f"This is the K_cols (Ordered): \n{K_csr.indices}\n\n")
    # print(f"This is the K_vals (Ordered): \n{K_csr.data}\n\n")


    # --- Timming the function --- #
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


    return K_csr, f









@cuda.jit
def K_csr_kernel(
        Ke, fe,
        Faces, K_indptr,
        intFaces, extFaces,
        nNodesPerFace, nIntFaces, nFacesPerEl,
        K_cols, K_vals, f,          # Output
        isIntFace=True              # Flag to have use this kernel for internal and external faces
        ):
    """
    This kernel creates the values and columns vector for the K_csr matrix in parallel.

    Note: Need to be run twice, one for intFaces and one for extFaces
    """
    # Shared memory of the local face of el1 to sum duplicate entries
    sh_localFaceEl1 = cuda.shared.array(1, int32)

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

    # Boolean flags to avoid repeated nodes (the ones on the sharing face)
    isSharedFace = (localFace == localNeighborFace)   # Is this the shared face
    isRepeated = isSharedFace and (localEl == 1)      # We are not writing data of the second element
    needsFaceCorrection = (localNeighborFace > localFace) and (localEl == 1) # Shifts the csrEntries to compensate for the repeated face

    # Write face data into the shared array and retrieve for repeated face
    if (isSharedFace) and (localEl == 0) and (localNeighborNode == 0):
        sh_localFaceEl1[0] = localFace
    cuda.syncthreads()

    correctLocalFace = sh_localFaceEl1[0] * isRepeated

    # Row to write in the CSR matrix
    if isRepeated: # Exceptions for the repeated face
        localCsrRow = localNeighborNode + correctLocalFace*nNodesPerFace
    else:
        localCsrRow = localNodeColumn + localEl*nNodesPerFace*(nFacesPerEl - needsFaceCorrection)


    # For loop with the number of face nodes
    for faceNode in range(nNodesPerFace):
        # Get the data from the stiffness matrix
        localNodeRow = faceNode + localFace*nNodesPerFace
        K_val = Ke[globalEl, localNodeRow, localNodeColumn]
        f_val = fe[globalEl, localNodeColumn]

        # Obtaian the row to write the data in the CSR vectors
        globalNode = faceNode + globalFace*nNodesPerFace 
        csrOffset = K_indptr[globalNode]
        globalCsrRow = localCsrRow + csrOffset

        # Write the values into the CSR vector
        if isSharedFace:    # Handle duplicate nodes
            cuda.atomic.add(K_vals, globalCsrRow, K_val)
        else:
            K_vals[globalCsrRow] = K_val

        # No need to write exception for K_vals (benign race condition)
        K_cols[globalCsrRow] = globalNeighborNode

    # Only write data from own face (to avoid duplicate data)
    if localFace == localNeighborFace:
        cuda.atomic.add(f, globalNeighborNode, f_val)



