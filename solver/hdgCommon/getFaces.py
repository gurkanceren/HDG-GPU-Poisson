import numpy as np


def get_faces(Conn):
    """
    Constructs internal and external face connectivity matrices for a 2D triangle mesh.
    
    Parameters:
        Conn (np.ndarray): Connectivity matrix of shape (nEl, 3), 0-indexed.
        
    Returns:
        intFaces (np.ndarray): Array of shape (N_int, 5). 
                               [el1, lf1, el2, lf2, match_node]
        extFaces (np.ndarray): Array of shape (N_ext, 2). 
                               [el, lf]

    Notation: 
        el1: First element connecting to the face (lowest index)
        lf1: Local face of el1 that corresponds to this global face
        el2: Second element connecting to the face (highest index)
        lf2: Local face of el2 that corresponds to this global face
        match_node: Global node of the first node of the face from the first element

    """
    nEl = Conn.shape[0]
    
    # Extract all local faces (edges) ordered by element
    face_nodes = np.empty((nEl, 3, 2), dtype=Conn.dtype)
    face_nodes[:, 0, :] = Conn[:, [0, 1]]
    face_nodes[:, 1, :] = Conn[:, [1, 2]]
    face_nodes[:, 2, :] = Conn[:, [2, 0]]
    face_nodes = face_nodes.reshape(-1, 2)

    # Track the element and local face index for every face
    elements = np.repeat(np.arange(nEl), 3)
    local_face_idx = np.tile([0, 1, 2], nEl)
    
    # Sort nodes internally to standardise the face representation
    sorted_faces = np.sort(face_nodes, axis=1)
    
    # Identify duplicates to group internal vs external faces
    _, inverse_idx, counts = np.unique(sorted_faces, axis=0, return_inverse=True, return_counts=True)
    
    # Masks
    ext_mask = counts[inverse_idx] == 1
    int_mask = counts[inverse_idx] == 2
    
    # --- Assemble extFaces ---
    extFaces = np.column_stack((elements[ext_mask], local_face_idx[ext_mask]))
    
    # --- Assemble intFaces ---
    int_idx = np.where(int_mask)[0]
    
    # Sort the internal face indices by their unique global identifier to pair them up
    couple_order = np.argsort(inverse_idx[int_idx])
    coupled_idx = int_idx[couple_order]
    coupled_idx_reshape = coupled_idx.reshape(coupled_idx.shape[0] // 2, 2)
    coupled_sorted_idx = np.sort(coupled_idx_reshape, axis=1)
    
    # Extract element 1 (even indices) and element 2 (odd indices) for each shared face
    idx1 = coupled_sorted_idx[:,0]
    idx2 = coupled_sorted_idx[:,1]
    
    el1 = elements[idx1]
    lf1 = local_face_idx[idx1]
    el2 = elements[idx2]
    lf2 = local_face_idx[idx2]
    
    # The matching node is the global node of the first node of the face from the first element
    match_node = face_nodes[idx1, 0]
    
    intFaces = np.column_stack((el1, lf1, el2, lf2, match_node)).astype(np.int32)
    extFaces = extFaces.astype(np.int32)

    # print("\n--------- Conn ---------")
    # print(f"{Conn}\n")
    # print("\n--------- IntFaces ---------")
    # print("### Old ###")
    # print(f"{intFaces_old}\n")
    # print("### New ###")
    # print(f"{intFaces}\n")
    # print("\n--------- ExtFaces ---------")
    # print("### Old ###")
    # print(f"{extFaces_old}\n")
    # print("### New ###")
    # print(f"{extFaces}\n")
    
    return intFaces, extFaces
