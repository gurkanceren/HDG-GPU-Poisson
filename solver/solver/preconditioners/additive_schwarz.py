import time
import math

import cupy as cp
from numba import cuda

from ..K_matrix import assemble_K_csr

def create_ASM_preconditioner(
        Ke, fe,
        Faces, intFaces, extFaces,
        nEl, nNodesPerFace, nFacesPerEl, nIntFaces,
        ):
    # Data types
    intDtype = Faces.dtype
    floatDtype = Ke.dtype

    # Modify the already created array to avoid copy costs
    # We are not using Ke more, so this should be fine
    Pe = Ke.copy()

    # Reshape matrix to have clearer indexing
    Ke = Ke.reshape(nEl, nFacesPerEl, nNodesPerFace, nFacesPerEl, nNodesPerFace)
    Pe = Pe.reshape(nEl, nFacesPerEl, nNodesPerFace, nFacesPerEl, nNodesPerFace)

    # Run kernel to add the diagonal block enrichment
    """ 
    Cannot create a 2D grid of blocks of size (nFacesPerEl, nEl) 
        - The maximum dimension of the block.x is very high (2^31 - 1)
        - The maximum dimension of the block.y is low (65535)
        - To get coalescence, need to have block.y as the nEl
        - At refined mesh numbers, the nEl is higher than this block.y dimension
    """
    threads_per_block = (nNodesPerFace, nNodesPerFace)
    blocks_per_grid = nFacesPerEl * nEl
    diagonal_enrichment_kernel[blocks_per_grid, threads_per_block](
            Pe, Ke,
            Faces, intFaces,
            nEl, nNodesPerFace, nFacesPerEl, nIntFaces,
            )

    # Invert using batched inversion
    Pe = Pe.reshape(nEl, nFacesPerEl*nNodesPerFace, nFacesPerEl*nNodesPerFace)
    Pe_inv = cp.linalg.inv(Pe)

    # Assemble in CSR format
    Pe_csr, _ = assemble_K_csr(
                       Pe_inv, fe,
                       Faces,
                       intFaces, extFaces,
                       nNodesPerFace, nFacesPerEl,
                       intDtype, floatDtype
                       )

    return Pe_csr



@cuda.jit
def diagonal_enrichment_kernel(
        Pe, Ke,
        Faces, intFaces,
        nEl, nNodesPerFace, nFacesPerEl, nIntFaces,
        ):

    # Identify threads
    localNodeCol = cuda.threadIdx.x
    localNodeRow = cuda.threadIdx.y
    localFace = cuda.blockIdx.x % nFacesPerEl
    el = cuda.blockIdx.x // nFacesPerEl

    # Bounds 
    if (localFace < nFacesPerEl) and (el < nEl):
        # Obtain the global face indices for the element
        globalFace = Faces[el, localFace]

        # Only enrich interior faces (exterior faces don't have additional contribution)
        if globalFace < nIntFaces:
            # Obtain neighbor element and local face 
            # This should be fast because we retrieve all 4 values at once???
            el1   = intFaces[globalFace, 0]
            face1 = intFaces[globalFace, 1]
            el2   = intFaces[globalFace, 2]
            face2 = intFaces[globalFace, 3]

            if el1 == el:
                neighborEl = el2
                neighborLocalFace = face2
            else:
                neighborEl = el1
                neighborLocalFace = face1

            # Enrich diagonal
            neighborPe = Ke[neighborEl, 
                            neighborLocalFace, localNodeRow, 
                            neighborLocalFace, localNodeCol]
            # cuda.syncthreads()

            Pe[el, localFace, localNodeRow, localFace, localNodeCol] += neighborPe





