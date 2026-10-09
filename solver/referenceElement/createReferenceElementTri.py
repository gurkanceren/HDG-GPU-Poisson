import numpy as np

import scipy.io as sio

from referenceElement.Gauss_Legendre_2d import gauss_legendre_cubature_2d

from referenceElement.gaussLegendre import gauss_legendre

from referenceElement.evaluateNodalBasisTri import evaluate_nodal_basis_tri

from referenceElement.evaluateNodalBasis1D import evaluate_nodal_basis_1D

from referenceElement.feketeNodesDict import fekete_nodes_1D

from referenceElement.convertMatToNpz import mat_to_npz

from referenceElement.convertMatToNpz import print_npz

def create_reference_element_tri(n_deg):
    """
    Creates the reference element for triangular elements.
    
    Output:
    theReferenceElement: struct containing
        .IPcoordinates: coordinates of the integration points for 2D elemens
        .IPweights: weights of the integration points for 2D elements
        .N: shape functions at the IP
        .Nxi,.Neta: derivatives of the shape functions at the IP
        .IPcoordinates1d: coordinates of the integration points for 1D boundary elemens
        .IPweights1d: weights of the integration points for 1D boundary elements
        .N1d: 1D shape functions at the IP
        .N1dxi: derivatives of the 1D shape functions at the IP
        .faceNodes: matrix [nOfFaces nOfNodesPerFace] with the edge nodes numbering
        .innerNodes: vector [1 nOfInnerNodes] with the inner nodes numbering
        .faceNodes1d: vector [1 nOfNodesPerElement] with the 1D nodes numbering
        .NodesCoord: spatial coordinates of the element nodes
        .NodesCoord1d: spatial coordinates of the 1D element nodes
    """
    face_nodes, inner_nodes, face_nodes_1d, coord_2d, coord_1d = None, None, None, None, None
    
    if n_deg == 1:
        face_nodes = np.array([[0, 1], [1, 2], [2, 0]])
        inner_nodes = []
        face_nodes_1d = np.array([0, 1])
        coord_2d = np.array([[-1, -1], [1, -1], [-1, 1]])
        coord_1d = np.array([[-1], [1]])
    elif n_deg == 2:
        face_nodes = np.array([[0, 3, 1], [1, 4, 2], [2, 5, 0]])
        inner_nodes = []
        face_nodes_1d = np.array([0, 1, 2])
        coord_2d = np.array([[-1, -1], [1, -1], [-1, 1], [0, -1], [0, 0], [-1, 0]])
        coord_1d = np.array([[-1], [0], [1]])
    elif n_deg == 3:
        face_nodes = np.array([[0, 3, 4, 1], [1, 5, 6, 2], [2, 7, 8, 0]])
        inner_nodes = np.array([9])
        face_nodes_1d = np.array([0, 1, 2, 3])
    elif n_deg == 4:
        face_nodes = np.array([[0, 3, 4, 5, 1], [1, 6, 7, 8, 2], [2, 9, 10, 11, 0]])
        inner_nodes = np.arange(12, 15)
        face_nodes_1d = np.array([0, 1, 2, 3, 4])
    elif n_deg == 5:
        face_nodes = np.array([
            [0, 3, 4, 5, 6, 1],
            [1, 7, 8, 9, 10, 2],
            [2, 11, 12, 13, 14, 0]
        ])
        inner_nodes = np.arange(15, 21)
        face_nodes_1d = np.array([0, 1, 2, 3, 4, 5])
    else:
        raise ValueError("Error in reference element: element not yet implemented")

    # THIS NEEDED TO BE EXECUTED ONLY ONE TIME TO OBTAIN THE .NPZ FROM .MAT
    """
    mat_file = "C:/Users/Usuario/Desktop/TFM/TASK 1/HDG_Poisson_Python/T1_HDG_Poisson/referenceElement/positionFeketeNodesTri2D_EZ4U.mat" # Replace with the path to your .mat file
    npz_file = "C:/Users/Usuario/Desktop/TFM/TASK 1/HDG_Poisson_Python/T1_HDG_Poisson/referenceElement/positionFeketeNodesTri2D_EZ4U.npz"   # Replace with the desired .npz output path
    mat_to_npz(mat_file, npz_file)
    print_npz(npz_file)"""

    # PREVIOUS LINES MAY BE WRONG, TRYING SMTH NEW BELOW:

    mat_file = "referenceElement/positionFeketeNodesTri2D_EZ4U.mat"
    data_mat = sio.loadmat(mat_file)  
    
    fekete_struct = data_mat["feketeNodesPosition"]  
    if n_deg >= 3:
        clave = f"P{n_deg}"  
        coord_2d = fekete_struct[0, 0][clave]  
        coord_1d = fekete_nodes_1D(n_deg, face_nodes_1d)

    # Set quadrature order
    quadrature_orders = {1: 5, 2: 10, 3: 10, 4: 15, 5: 15, 6: 15, 7: 15, 8: 25, 9: 25, 10: 25, 11: 25}
    order_cubature = quadrature_orders.get(n_deg, None)
    
    # Compute integration points and weights
    z, w = gauss_legendre_cubature_2d(order_cubature)
    gp_2d = 2 * z - 1  # Mapping to reference triangle
    gw_2d = 2 * w # in matlab is trasposed <----------------------------------
   
    n_of_gauss_points_1d = 2 * n_deg + 1
    gp_1d, gw_1d = gauss_legendre(n_of_gauss_points_1d)
 
    
    # Compute shape functions and derivatives
    N, Nxi, Neta = evaluate_nodal_basis_tri(gp_2d, coord_2d, n_deg)
    N1d, Nxi1d = evaluate_nodal_basis_1D(gp_1d, coord_1d, n_deg)
    
    return {
        "IPcoordinates": gp_2d,
        "IPweights": gw_2d,
        "N": N,
        "Nxi": Nxi,
        "Neta": Neta,
        "IPcoordinates1d": gp_1d,
        "IPweights1d": gw_1d,
        "N1d": N1d,
        "N1dxi": Nxi1d,
        "faceNodes": face_nodes,
        "innerNodes": inner_nodes,
        "faceNodes1d": face_nodes_1d,
        "NodesCoord": coord_2d,
        "NodesCoord1d": coord_1d,
        "degree": n_deg
    }
