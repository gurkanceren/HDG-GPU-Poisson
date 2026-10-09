import time
from math import ceil
import numpy as np
import cupy as cp
from numba import cuda

# from hdgCommon.getFaces import get_faces
from hdgCommon.get_faces_gpu import get_faces
from hdgCommon.neighbors import obtain_neighbors

def hdg_preprocess(Conn):
    """
    Processes the mesh connectivity matrix Conn to create face connectivity Faces 
    and identify internal and external faces, using 0-indexed values.
    
    Parameters:
        Conn (numpy.ndarray): Mesh connectivity matrix (Mx3 for triangular elements), 0-indexed.
    
    Returns:
        Faces (numpy.ndarray): Face connectivity matrix (Mx3) with 0-indexed face IDs.
        infoFaces (dict): Dictionary containing 'intFaces' and 'extFaces',
                          with entries in 0-indexed format:
                          - intFaces rows: [element1, face1, element2, face2, common_node]
                          - extFaces rows: [element, face]
    """
    # Get internal and external faces using the first 3 columns of Conn.
    tic_0 = time.perf_counter()
    # (Assume get_faces now returns 0-indexed information.)
    intFaces, extFaces = get_faces(Conn[:, :3])
    nIntFaces = intFaces.shape[0]   
    nExtFaces = extFaces.shape[0]   
    print("Get Faces completed")
    
    # Number of elements.
    tic_1 = time.perf_counter()
    nEl = Conn.shape[0]
    
    # Initialize face connectivity matrix Faces (each element has 3 faces).
    Faces = cp.zeros((nEl, 3), dtype=cp.int32)
    
    # Assign internal faces: enumerate from 0.
    tic_2 = time.perf_counter()
    threads_per_block = 256
    blocks_per_grid = ceil(nIntFaces / threads_per_block)
    nElPerFace = 2
    offset = 0
    assign_faces[blocks_per_grid, threads_per_block](
            intFaces,       # Array of interior or exterior faces
            Faces,          # Output array
            nIntFaces,      # Number of internal or external faces
            nElPerFace,   # Number of elements sharing the face: 2 for internal, 1 for external
            offset,       # Offset to apply to the indices of the faces
            )
    print("Assign Internal Faces completed")
    
    # Assign external faces: enumerate from 0, but offset by the number of internal faces.
    tic_3 = time.perf_counter()
    threads_per_block = 256
    blocks_per_grid = ceil(nExtFaces / threads_per_block)
    nElPerFace = 1
    offset = nIntFaces
    assign_faces[blocks_per_grid, threads_per_block](
            extFaces,       # Array of interior or exterior faces
            Faces,          # Output array
            nExtFaces,      # Number of internal or external faces
            nElPerFace,   # Number of elements sharing the face: 2 for internal, 1 for external
            offset,       # Offset to apply to the indices of the faces
            )
    print("Assign External Faces completed")
    
    # Store the face information in a dictionary.
    tic_4 = time.perf_counter()
    infoFaces_dict = {
        "intFaces": intFaces,
        "extFaces": extFaces
    }

    tic_5 = time.perf_counter()

    # Create matrix of neighbors (for the Linear Operator of CG)
    threads_per_block = 128
    blocks_per_grid = ceil(nIntFaces / threads_per_block)
    nNeighbors = 5  # For 2D triangles

    Faces_neighbors_gpu = cp.zeros((nIntFaces, nNeighbors), dtype=cp.int32)
    Faces_gpu = cp.asarray(Faces)
    intFaces_gpu = cp.asarray(intFaces)

    obtain_neighbors[blocks_per_grid, threads_per_block](Faces_gpu, intFaces_gpu, nIntFaces, Faces_neighbors_gpu)
    cuda.synchronize()

    tic_6 = time.perf_counter()

    # Print the times of each function
    total_time = tic_6 - tic_0
    percent = lambda t: t / total_time * 100

    # print(f"\nEvaluating the Pre-processing")
    # print(f"\t \t Part 1: Get faces")
    # print(f"\t \t Taken up {percent(tic_1 - tic_0):.1f}% ({tic_1 - tic_0:.3e}s)")
    # print(f"\t \t Part 2: Initialize Faces matrix")
    # print(f"\t \t Taken up {percent(tic_2 - tic_1):.1f}% ({tic_2 - tic_1:.3e}s)")
    # print(f"\t \t Part 3: Assign internal faces")
    # print(f"\t \t Taken up {percent(tic_3 - tic_2):.1f}% ({tic_3 - tic_2:.3e}s)")
    # print(f"\t \t Part 4: Assign external faces")
    # print(f"\t \t Taken up {percent(tic_4 - tic_3):.1f}% ({tic_4 - tic_3:.3e}s)")
    # print(f"\t \t Part 5: Store face information in dictionary")
    # print(f"\t \t Taken up {percent(tic_5 - tic_4):.1f}% ({tic_5 - tic_4:.3e}s)")
    # print(f"\t \t Part 6: Create matrix of neighbours")
    # print(f"\t \t Taken up {percent(tic_6 - tic_5):.1f}% ({tic_6 - tic_5:.3e}s)")
    # print(f"\n \t \t Function took {total_time:.4f}s\n \n")

    return Faces, infoFaces_dict, Faces_neighbors_gpu







# Function to assign the values to the Faces matrix
@cuda.jit
def assign_faces(
        intExtFaces,    # Array of interior or exterior faces
        Faces,          # Output array
        nIntExtFaces,   # Number of internal or external faces
        nElPerFace,     # Number of elements sharing the face: 2 for internal, 1 for external
        offset,         # Offset to apply to the indices of the faces
        ):

    # Identify threads
    intExtFace = cuda.grid(1)

    # Bounds
    if (intExtFace < nIntExtFaces):
        # Apply offset to the index of the face
        faceIdx = intExtFace + offset

        # Introduce values of the elements
        for faceEl in range(nElPerFace):
            el = intExtFaces[intExtFace, faceEl*2]
            localFace = intExtFaces[intExtFace, faceEl*2 + 1]

            Faces[el, localFace] = faceIdx







