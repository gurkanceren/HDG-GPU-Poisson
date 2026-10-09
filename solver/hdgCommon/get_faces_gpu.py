import time

import cupy as cp



def get_faces(Conn, intDtype=cp.int32):
    """
    Constructs internal and external face connectivity matrices for a 2D triangle mesh.
    
    Parameters:
        Conn (cp.ndarray): Connectivity matrix of shape (nEl, 3), 0-indexed.
        
    Returns:
        intFaces (cp.ndarray): Array of shape (N_int, 4). 
                               [el1, lf1, el2, lf2]
        extFaces (cp.ndarray): Array of shape (N_ext, 2). 
                               [el, lf]

    Notation: 
        el1: First element connecting to the face (lowest index)
        lf1: Local face of el1 that corresponds to this global face
        el2: Second element connecting to the face (highest index)
        lf2: Local face of el2 that corresponds to this global face

    """
    tic_0 = time.perf_counter()
    nEl = Conn.shape[0]
    nFacesPerEl = 3
    Conn = cp.asarray(Conn)


    # Create intermediate arrays needed for the faces
    tic_1 = time.perf_counter()
    faceNodes = cp.empty((nEl, nFacesPerEl, 4), dtype=intDtype, order="C")

    connectionFaces = cp.arange(nFacesPerEl + 1)    # How faces connect
    connectionFaces[-1] = 0                         # For a triangle [0, 1, 2, 0]

    elementIdx = cp.arange(nEl)


    # Assemble array of faces (including duplicates)
    # This array will be composed of [nodeIdxMin, nodeIdxMax, elementIdx, localFaceIdx]
    tic_2 = time.perf_counter()
    for localFace in range(nFacesPerEl):
        node1 = connectionFaces[localFace]
        node2 = connectionFaces[localFace + 1]

        faceNodes[:, localFace, 0] = cp.minimum(Conn[:, node1], Conn[:, node2])
        faceNodes[:, localFace, 1] = cp.maximum(Conn[:, node1], Conn[:, node2])
        faceNodes[:, localFace, 2] = elementIdx
        faceNodes[:, localFace, 3] = localFace

    faceNodes = faceNodes.reshape(nEl*nFacesPerEl, 4)


    # Obtain the indices that sort faces in this order -> MinNode, MaxNode, el
    # This groups the repeated faces togheter and keeps the lower index element first
    # TODO: See how much the element sorting affects performance to remove it from here or not
    tic_3 = time.perf_counter()
    sortedIdx = cp.lexsort(cp.vstack((faceNodes[:, 2], faceNodes[:, 1], faceNodes[:, 0])))


    # Sort the array with the indices
    tic_4 = time.perf_counter()
    sortedFaceNodes = faceNodes[sortedIdx]

    
    # Compare each row to the next to find the duplicate faces
    tic_5 = time.perf_counter()
    intMask = (sortedFaceNodes[:-1, 0] == sortedFaceNodes[1:, 0]) & \
              (sortedFaceNodes[:-1, 1] == sortedFaceNodes[1:, 1])

    isInterior = cp.zeros(nEl*nFacesPerEl, dtype=bool) 
    isInterior[:-1] = intMask
    isInterior[1:] = cp.logical_or(intMask, isInterior[1:])
    extMask = ~isInterior
    

    # Extract indices where first element of a pair occurs
    tic_6 = time.perf_counter()
    intIdx = cp.nonzero(intMask)[0]
    

    # Assemble intFaces
    tic_7 = time.perf_counter()
    nIntFaces = intIdx.size
    intFaces = cp.empty((nIntFaces, 4), dtype=intDtype)

    intFaces[:, 0] = sortedFaceNodes[intIdx, 2]
    intFaces[:, 1] = sortedFaceNodes[intIdx, 3]
    intFaces[:, 2] = sortedFaceNodes[intIdx + 1, 2]
    intFaces[:, 3] = sortedFaceNodes[intIdx + 1, 3]


    
    # Assemble extFaces
    tic_8 = time.perf_counter()
    extFaces = sortedFaceNodes[extMask, 2:]
    
    # Change to numpy for rest of code
    tic_9 = time.perf_counter()
    # intFaces = cp.asnumpy(intFaces)
    # extFaces = cp.asnumpy(extFaces)

    # Print the times of each function
    tic_10 = time.perf_counter()
    total_time = tic_10 - tic_0
    percent = lambda t: t / total_time * 100

    # print(f"\nEvaluating the getFaces function")
    # print(f"\t \t Part 1: Move Conn to GPU")
    # print(f"\t \t Taken up {percent(tic_1 - tic_0):.1f}% ({tic_1 - tic_0:.3e}s)")
    # print(f"\t \t Part 2: Create auxiliary arrays")
    # print(f"\t \t Taken up {percent(tic_2 - tic_1):.1f}% ({tic_2 - tic_1:.3e}s)")
    # print(f"\t \t Part 3: Assemble array of faces")
    # print(f"\t \t Taken up {percent(tic_3 - tic_2):.1f}% ({tic_3 - tic_2:.3e}s)")
    # print(f"\t \t Part 4: Obtain the sorting indices")
    # print(f"\t \t Taken up {percent(tic_4 - tic_3):.1f}% ({tic_4 - tic_3:.3e}s)")
    # print(f"\t \t Part 5: Sort the faces array")
    # print(f"\t \t Taken up {percent(tic_5 - tic_4):.1f}% ({tic_5 - tic_4:.3e}s)")
    # print(f"\t \t Part 6: Create the masks")
    # print(f"\t \t Taken up {percent(tic_6 - tic_5):.1f}% ({tic_6 - tic_5:.3e}s)")
    # print(f"\t \t Part 7: Extract internal indices")
    # print(f"\t \t Taken up {percent(tic_7 - tic_6):.1f}% ({tic_7 - tic_6:.3e}s)")
    # print(f"\t \t Part 8: Assemble intFaces")
    # print(f"\t \t Taken up {percent(tic_8 - tic_7):.1f}% ({tic_8 - tic_7:.3e}s)")
    # print(f"\t \t Part 9: Assemble extFaces")
    # print(f"\t \t Taken up {percent(tic_9 - tic_8):.1f}% ({tic_9 - tic_8:.3e}s)")
    # print(f"\t \t Part 10: Move data back to CPU")
    # print(f"\t \t Taken up {percent(tic_10 - tic_10):.1f}% ({tic_10 - tic_9:.3e}s)")
    # print(f"\n \t \t Function took {total_time:.4f}s\n \n")

    return intFaces, extFaces


