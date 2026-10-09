import time
import numpy as np

def get_faces(T):
    """
    Computes internal and external faces for a mesh of triangles.
    
    Input:
      T : Connectivity matrix of the mesh (shape: [nElements, 3] for triangles), 0-indexed.
    
    Output:
      intFaces, extFaces: 
         - intFaces: Array of internal faces, where each row is 
                     [element1, local_face_index1, element2, local_face_index2, common_node],
                     all 0-indexed.
         - extFaces: Array of external faces, each row is [element, local_face_index].
    """
    # Step 1: Dimensions and pre-allocation
    tic_0 = time.perf_counter()
    nElem, nen = T.shape              # nElem: number of elements; nen: nodes per element (here 3)
    nfaceel = nen                     # For a triangle, there are 3 faces.
    nNodes = int(np.max(T)) + 1        # Since T is 0-indexed

    # Create global connectivity matrix N (nNodes x 10) and counter vector nn
    N = np.zeros((nNodes, 10), dtype=int)
    nn = np.ones(nNodes, dtype=int)
    for ielem in range(nElem):         # ielem is 0-indexed
        Te = T[ielem, :]               # Connectivity for element ielem
        nn_Te = nn[Te].copy()          # Get counters for each node in the element
        for kk in range(3):            # For each of the 3 nodes in a triangle
            N[Te[kk], nn_Te[kk]-1] = ielem  # Store the element number (0-indexed)
        nn[Te] += 1
    max_nn = np.max(nn)
    N = N[:, :max_nn-1]                # Trim unused columns

    # Step 2: Initialize marking and face storage arrays
    tic_1 = time.perf_counter()
    markE = np.zeros((nElem, nfaceel), dtype=int)
    intFaces = np.zeros((int(1.5 * nElem), 5), dtype=int)
    extFaces = np.zeros((3 * nElem, 2), dtype=int)
    
    # Step 3: Define local face numbering (0-indexed)
    # For triangles: face 0: [0, 1], face 1: [1, 2], face 2: [2, 0]
    if nen == 3:
        Efaces = np.array([[0, 1],
                           [1, 2],
                           [2, 0]])
    elif nen == 4:
        raise NotImplementedError("Tetrahedra not implemented yet")
    else:
        raise ValueError("Unsupported element type.")
    
    # Step 4: Process each element and its faces
    tic_2 = time.perf_counter()
    intF = 0   # Internal face counter (0-indexed)
    extF = 0   # External face counter (0-indexed)
    for iElem in range(nElem):
        for iFace in range(nfaceel):
            if markE[iElem, iFace] == 0:
                markE[iElem, iFace] = 1
                # Extract nodes of the current face from element iElem
                nodesf = T[iElem, Efaces[iFace]]  # nodesf is already 0-indexed
                jelem = find_elem(iElem, nodesf, T, N)
                if jelem != -1:
                    jface, node1 = find_face(nodesf, T[jelem], Efaces)
                    # Ordena para que el elemento con menor índice venga primero
                    if iElem < jelem:
                        intFaces[intF, :] = [iElem, iFace, jelem, jface, node1]
                    else:
                        intFaces[intF, :] = [jelem, jface, iElem, iFace, node1]

                    intF += 1
                    markE[jelem, jface] = 1
                else:
                    extFaces[extF, :] = [iElem, iFace]
                    extF += 1
    intFaces = intFaces[:intF, :]
    extFaces = extFaces[:extF, :]


    tic_3 = time.perf_counter()

    # Print the times of each function
    total_time = tic_3 - tic_0
    percent = lambda t: t / total_time * 100

    print(f"\nEvaluating the GetFaces function")
    print(f"\t \t Part 1: Taken up {percent(tic_1 - tic_0):.1f}% ({tic_1 - tic_0:.3e}s)")
    print(f"\t \t Part 2: Taken up {percent(tic_2 - tic_1):.1f}% ({tic_2 - tic_1:.3e}s)")
    print(f"\t \t Part 3: Taken up {percent(tic_3 - tic_2):.1f}% ({tic_3 - tic_2:.3e}s)")
    print(f"\n \t \t Function took {total_time:.4f}s\n \n")
    
    return intFaces, extFaces

def find_elem(iElem, nodesf, T, N):
    """
    Finds a neighboring element (jelem) that shares the face defined by nodesf with the current element iElem.
    Operates with 0-indexed numbers.
    
    Parameters:
        iElem  : Current element index (0-indexed).
        nodesf : Array of nodes (0-indexed) defining the face of element iElem.
        T      : Connectivity matrix (0-indexed).
        N      : Global connectivity matrix from get_faces (0-indexed).
    
    Returns:
        jelem : The index (0-indexed) of the neighboring element sharing the face, or -1 if none is found.
    """
    nen_local = len(nodesf)
    # Use the first node of the face to get candidate elements
    elems = N[nodesf[0], N[nodesf[0], :] != 0]
    elems = elems[elems != iElem]  # Exclude the current element
    if elems.size == 0:
        return -1
    Ti = T[elems]  # Connectivity for candidate elements
    for i in range(1, nen_local):
        if elems.size > 0:
            mask = np.any(Ti == nodesf[i], axis=1)
            elems = elems[mask]
            Ti = Ti[mask]
    if elems.size == 0:
        return -1
    else:
        return int(elems[0])

def find_face(nodesf, nodesE, Efaces):
    """
    In the neighboring element with connectivity nodesE, finds which local face matches the face defined by nodesf.
    All indices are 0-indexed.
    
    Parameters:
        nodesf : Array of nodes (0-indexed) for the current element's face.
        nodesE : Connectivity of the neighboring element (0-indexed).
        Efaces : Local face definitions (0-indexed) for the reference element.
    
    Returns:
        jface : The local face number (0-indexed) in the neighboring element that matches nodesf.
        node1 : The common node (0-indexed) in the neighboring element.
    """
    nFaces = Efaces.shape[0]
    jface = -1
    node1 = -1
    for j in range(nFaces):
        nodesj = nodesE[Efaces[j]]  # Nodes for the j-th face of the neighboring element
        # Check if nodesj and nodesf share the same nodes (order does not matter)
        if ((nodesj[0] == nodesf[0] or nodesj[0] == nodesf[1]) and
            (nodesj[1] == nodesf[0] or nodesj[1] == nodesf[1])):
            jface = j
            # Mimic MATLAB: choose nodesf[0] if present, otherwise nodesf[1]
            if nodesf[0] in nodesj:
                node1 = nodesf[0]
            else:
                node1 = nodesf[1]
            break
    return jface, node1
