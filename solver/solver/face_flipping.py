from numba import cuda
import cupy as cp
import numpy as np

@cuda.jit
def face_flipping_kernel(
        UU, QQ, Alq, Alu, All,          # Matrices to be flipped
        intFaces_gpu, Faces,            
        nIntFaces, nFacesPerEl, nEl, nNodesPerEl, nDim, nNodesPerFace
        ):
    """
    Kernel that flips the faces of the matrices to be consistent across elements.
    Since all elements count nodes c.c.w., when looking at a face F, the nodes are numbered as:
        - Element 1: N1, N2, ..., NNodesPerFace
        - Element 2: NNodesPerFace, ..., N2, N1
    Ne
    """

    # Thread identification
    localFace = cuda.threadIdx.x
    e = cuda.blockIdx.x

    # Bounds 
    if (localFace < nFacesPerEl) and (e < nEl):
        
        # Get global face and determine if it is internal
        globalFace = Faces[e, localFace]
        isInteriorFace = (globalFace < nIntFaces)

        if isInteriorFace:
            # Check if it the face needs to be flipped
            # We are flipping only the second faces on the array (higher element number)
            needsFlipping = (e == intFaces_gpu[globalFace, 2])

            if needsFlipping:

                # Custom loop to avoid Cupy interactions
                # Only loop over half the nodes to avoid double-substitution
                halfNodes = nNodesPerFace // 2

                for faceNode in range(halfNodes):
                    reverseNode = nNodesPerFace - 1 - faceNode 

                    # Loop over the "u" and "q" variable entries
                    for uNode in range(nNodesPerEl):

                        # Swap variables with "u" dimensions
                        temp_UU = UU[e, uNode, localFace, faceNode]
                        UU[e, uNode, localFace, faceNode] = UU[e, uNode, localFace, reverseNode]
                        UU[e, uNode, localFace, reverseNode] = temp_UU

                        temp_Alu = Alu[e, localFace, faceNode, uNode]
                        Alu[e, localFace, faceNode, uNode] = Alu[e, localFace, reverseNode, uNode]
                        Alu[e, localFace, reverseNode, uNode] = temp_Alu

                        # Swap variables with "q" dimensions
                        for dim in range(nDim):
                            # New index of the "q" variable
                            qNode = uNode*nDim + dim

                            temp_QQ = QQ[e, qNode, localFace, faceNode]
                            QQ[e, qNode, localFace, faceNode] = QQ[e, qNode, localFace, reverseNode]
                            QQ[e, qNode, localFace, reverseNode] = temp_QQ

                            temp_Alq = Alq[e, localFace, faceNode, qNode]
                            Alq[e, localFace, faceNode, qNode] = Alq[e, localFace, reverseNode, qNode]
                            Alq[e, localFace, reverseNode, qNode] = temp_Alq

                    # NOTE: For the All, it seems to be symmetric, so we do not need flipping
                    # Need to check if it is only for this case or for every other one

                    # # Loop over the "l" variable entries
                    # # Flip the faces in one direction
                    # for lNode in range(nNodesPerEl):
                    #
                    #     temp_All = All[e, lNode, localFace, localFace, faceNode]
                    #     All[e, lNode, localFace, localFace, faceNode] = All[e, lNode, localFace, localFace, reverseNode]
                    #     All[e, lNode, localFace, localFace, reverseNode] = temp_All
                    #
                    # # Flip the faces in the other direction
                    # for lNode in range(nNodesPerEl):
                    #
                    #     temp_All = All[e, faceNode, localFace, localFace, lNode]
                    #     All[e, faceNode, localFace, localFace, lNode] = All[e, reverseNode, localFace, localFace, lNode]
                    #     All[e, reverseNode, localFace, localFace, lNode] = temp_All




    
# Test bench
if __name__ == "__main__":
    # Define parameters
    nEl = 1
    nNodesPerFace = 3
    nFacesPerEl = 1
    nNodesPerEl = 3

    e = 0
    localFace = 0

    # Create easy matrix
    All = cp.arange(9) + 1
    All = All.reshape(nEl, nNodesPerFace * nFacesPerEl, nNodesPerFace * nFacesPerEl)
    print(f"This is the All matrix before flipping: \n {All}\n\n")
    All = All.reshape(nEl, nNodesPerFace, nFacesPerEl, nNodesPerFace, nFacesPerEl)


    # Custom loop to avoid Cupy interactions
    # Only loop over half the nodes to avoid double-substitution
    halfNodes = nNodesPerEl // 2

    for faceNode in range(halfNodes):
        reverseNode = nNodesPerFace - 1 - faceNode 

        # Flip the faces in one direction
        for lNode in range(nNodesPerFace):

            temp_All = All[e, lNode, localFace, faceNode, localFace].item()
            All[e, lNode, localFace, faceNode, localFace] = All[e, lNode, localFace, reverseNode, localFace]
            All[e, lNode, localFace, reverseNode, localFace] = temp_All

        # Flip the faces in the other direction
        for lNode in range(nNodesPerFace):

            temp_All = All[e, faceNode, localFace, lNode, localFace].item()
            All[e, faceNode, localFace, lNode, localFace] = All[e, reverseNode, localFace, lNode, localFace]
            All[e, reverseNode, localFace, lNode, localFace] = temp_All

    cuda.synchronize()
    All = All.reshape(nEl, nNodesPerFace * nFacesPerEl, nNodesPerFace * nFacesPerEl)
    print(f"This is the All matrix after flipping: \n {All}\n\n")
