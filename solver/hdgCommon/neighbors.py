from numba import cuda

@cuda.jit
def obtain_neighbors(F, intFaces, nIntFaces, F_neighbors):
    """
    Obtain the neighboring faces of a global face for the latter assembly into DBF


    """
    # Obtain thread data
    faceIdx = cuda.grid(1)

    nFacesPerEl = 3 # For a 2D triangle

    # Guards for out of bound threads
    if faceIdx < nIntFaces:
        # Obtain face data
        el1 = intFaces[faceIdx, 0]
        localFace1 = intFaces[faceIdx, 1]
        el2 = intFaces[faceIdx, 2]
        localFace2 = intFaces[faceIdx, 3]

        # Search on the element matrix for neighbor faces
        allFaces1 = F[el1, :]
        allFaces2 = F[el2, :]

        # Include the global face number in first place
        globalFace = allFaces1[localFace1]
        F_neighbors[faceIdx, 0] = globalFace 

        # Neighboring faces of the elements
        neighborIdx = 1
        for i in range(nFacesPerEl):
            # Element 1
            neighborFace1 = allFaces1[i]
            if (neighborFace1 != globalFace) and (neighborFace1 < nIntFaces):
                F_neighbors[faceIdx, neighborIdx] = neighborFace1
                neighborIdx += 1

            # Element 2
            neighborFace2 = allFaces2[i]
            if (neighborFace2 != globalFace) and (neighborFace2 < nIntFaces):
                F_neighbors[faceIdx, neighborIdx] = neighborFace2
                neighborIdx += 1

        # Pad with "-1" for the boundary faces
        F_neighbors[faceIdx, neighborIdx:] = -1







