import numpy as np


def create_mesh(mesh_number, order):
    """
    This function creates a mesh based on the mesh number and the degree.

    Inputs:
        - mesh_number: It is an exponential parameter to determine the number of elements per row
    -> el_per_row = 2 [triangle elements per square] * 2^(mesh_number)
    - order: Polynomial degree of each elements

    Outputs:
        - NODES: Matrix with the 2D (or 3D) coordinates of all the NODES
    - CONN: Connectivity matrix for the elements (each row is an element)
    - Tb_Dirichlet: Boundary faces of the elements.
    """

    domainLength = 1
    nDim = 2  # Number of dimensions of the problem


    # ----------------- Important numbers for the mesh ----------------- #
    # Determine the number of nodes on each row (columns are the same as rows)
    nSquaresPerRow = 2**mesh_number
    nNodesPerRow = (nSquaresPerRow * order + 1)  # Each square has [order] independent nodes + last node of the row

    # Total number of elements and nodes
    nNodes = nNodesPerRow * nNodesPerRow

    # Additional numbers that will be useful
    nNodesTriFace = order + 1
    nNodesPerTriElem = sum([i for i in range(nNodesTriFace + 1)])
    nInternalNodes = max(0, nNodesPerTriElem - (3 * nNodesTriFace - 3))


    # Create the NODES, CONN, and BCs
    NODES = create_NODES(domainLength, nNodesPerRow, nDim, nNodes)
    CONN = create_CONN(nNodes, nNodesPerRow, nSquaresPerRow, nNodesTriFace, nInternalNodes, nNodesPerTriElem, order)
    Tb_Dirichlet = create_BCs(nSquaresPerRow, nNodesTriFace, nNodesPerRow)

    export_mesh(mesh_number, order, NODES, CONN, Tb_Dirichlet)



# ----------------- Create the NODES matrix ----------------- #
def create_NODES(domainLength, nNodesPerRow, nDim, nNodes):
    # Create the nodes coordinates
    auxCoordVector = np.linspace(0, domainLength, num=nNodesPerRow)
    auxCoordGrid = np.zeros((nNodesPerRow, nNodesPerRow, nDim))

    # For the moment, using meshgrid for the nodes
    auxCoordGrid[:, :, 0], auxCoordGrid[:, :, 1] = np.meshgrid(
        auxCoordVector, auxCoordVector, indexing="xy"
    )

    NODES = np.reshape(auxCoordGrid, (nNodes, nDim))

    return NODES

# ----------------- Create the CONN matrix ----------------- #
def create_CONN(nNodes, nNodesPerRow, nSquaresPerRow, nNodesTriFace, nInternalNodes, nNodesPerTriElem, order):

    # Auxiliary matrix with the nodes numbering
    auxNodes = np.arange(0, nNodes).reshape((nNodesPerRow, nNodesPerRow), order="F")
    step = order  # Number to step for each from square to square

    # Create matrices with the corners of each square
    BottomLeftCorners  = auxNodes[:-step:step, :-step:step]
    BottomRightCorners = auxNodes[step::step, :-step:step]
    TopLeftCorners     = auxNodes[:-step:step, step::step]
    TopRightCorners    = auxNodes[step::step, step::step]

    # Create matrices with the faces and internal nodes of each square
    # These are 3D matrices to count for adjacent nodes in the faces or internal nodes
    bottomFaceNodes = np.zeros((nSquaresPerRow, nSquaresPerRow, nNodesTriFace - 2))
    topFaceNodes    = np.zeros_like(bottomFaceNodes)
    leftFaceNodes   = np.zeros_like(bottomFaceNodes)
    rightFaceNodes  = np.zeros_like(bottomFaceNodes)
    diagFaceNodes   = np.zeros_like(bottomFaceNodes)

    botRightInternalNodes = np.zeros((nSquaresPerRow, nSquaresPerRow, nInternalNodes))
    topLeftInternalNodes = np.zeros_like(botRightInternalNodes)

    for face_nodes in range(nNodesTriFace - 2):
        start_node = 1 + face_nodes
        bottomFaceNodes[:, :, face_nodes] = auxNodes[start_node::step, :-order:step]
        topFaceNodes[:, :, face_nodes]    = auxNodes[start_node::step, order::step]
        leftFaceNodes[:, :, face_nodes]   = auxNodes[:-order:step, start_node::step]
        rightFaceNodes[:, :, face_nodes]  = auxNodes[order::step, start_node::step]
        diagFaceNodes[:, :, face_nodes]   = auxNodes[start_node::step, start_node::step]

    # The internal nodes have to be treated differently, we need to loop the diagonal nodes for both faces
    # TODO -> Vectorize or compile this part
    for row in range(nSquaresPerRow):
        for column in range(nSquaresPerRow):
            # Reset counters
            counterBotRight = 0
            counterTopLeft = 0

            for node_index in range(nNodesTriFace - 2):
                # Bottom right triangle
                node_diag = diagFaceNodes[row, column, node_index]
                node_face = rightFaceNodes[row, column, node_index]

                aux_range = np.arange(node_diag + 1, node_face)
                aux_lenght = len(aux_range)

                botRightInternalNodes[
                    row, column, counterBotRight : counterBotRight + aux_lenght
                ] = aux_range.copy()

                counterBotRight += aux_lenght

                # Top left triangle
                node_diag = diagFaceNodes[row, column, -1 - node_index]
                node_face = leftFaceNodes[row, column, -1 - node_index]

                aux_range = np.arange(node_diag - 1, node_face, step=-1)
                aux_lenght = len(aux_range)

                topLeftInternalNodes[
                    row, column, counterTopLeft : counterTopLeft + aux_lenght
                ] = aux_range.copy()

                counterTopLeft += aux_lenght


    # print(f"\n \n Bottom Left Nodes \n {BottomLeftCorners}")
    # print(f"\n \n Bottom Right Nodes \n {BottomRightCorners}")
    # print(f"\n \n Top Left Nodes \n {TopLeftCorners}")
    # print(f"\n \n Top Right Nodes \n {TopRightCorners}")
    #
    # print(f"\n \n Bottom Face Nodes \n {bottomFaceNodes}")
    # print(f"\n \n Top Face Nodes \n {topFaceNodes}")
    # print(f"\n \n Left Face Nodes \n {leftFaceNodes}")
    # print(f"\n \n Right Face Nodes \n {rightFaceNodes}")
    # print(f"\n \n Diagonal Face Nodes \n {diagFaceNodes}")
    # print(f"\n \n Internal Nodes \n {botRightInternalNodes}")
    # print(f"\n \n Internal Nodes \n {topLeftInternalNodes}")

    # Put the Connectivity
    # Start the connectivity as 3D for simplicity
    CONN = np.zeros((nSquaresPerRow, 2 * nSquaresPerRow, nNodesPerTriElem), dtype=np.int64)
    for row in range(0, nSquaresPerRow, 1):
        for column in range(0, nSquaresPerRow, 1):
            # Assemble the bottom right triangle
            CONN[row, column, :] = np.hstack(
                (
                    BottomLeftCorners[row, column],
                    BottomRightCorners[row, column],
                    TopRightCorners[row, column],
                    bottomFaceNodes[row, column, :],
                    rightFaceNodes[row, column, :],
                    np.flip(diagFaceNodes[row, column, :]),
                    botRightInternalNodes[row, column, :],
                )
            )

            # Assemble the top left triangle
            CONN[row, column + nSquaresPerRow, :] = np.hstack(
                (
                    TopRightCorners[row, column],
                    TopLeftCorners[row, column],
                    BottomLeftCorners[row, column],
                    np.flip(topFaceNodes[row, column, :]),
                    np.flip(leftFaceNodes[row, column, :]),
                    diagFaceNodes[row, column, :],
                    topLeftInternalNodes[row, column, :],
                )
            )

    # Reshape CONN to final form
    CONN = CONN.reshape((2 * nSquaresPerRow**2, nNodesPerTriElem))
    # print(f"\n \n CONN \n {CONN}")

    return CONN

# ----------------- Create the BCs ----------------- #
def create_BCs(nSquaresPerRow, nNodesTriFace, nNodesPerRow):
    nSides = 4
    nBC = nSides * nSquaresPerRow  # One BC per row of squares on each side

    Tb_Dirichlet = np.zeros((nBC, nNodesTriFace), dtype=np.int64)
    
    # Create auxiliary vector to get the face nodes
    auxTbVector = np.arange(0, nSquaresPerRow * order, step=order)

    # TODO: This needs cleanup.
    # Bottom face
    sideIdx = 0
    startRow = sideIdx * nSquaresPerRow
    endRow = (sideIdx + 1) * nSquaresPerRow

    for col in range(nNodesTriFace):
        Tb_Dirichlet[startRow:endRow, col] = auxTbVector + col

    # Top face
    sideIdx = 1
    startRow = sideIdx * nSquaresPerRow
    endRow = (sideIdx + 1) * nSquaresPerRow
    offset = nNodesPerRow * (nNodesPerRow - 1)

    for col in range(nNodesTriFace):
        Tb_Dirichlet[startRow:endRow, col] = auxTbVector + col + offset

    # Left face
    sideIdx = 2
    startRow = sideIdx * nSquaresPerRow
    endRow = (sideIdx + 1) * nSquaresPerRow

    for col in range(nNodesTriFace):
        Tb_Dirichlet[startRow:endRow, col] = (auxTbVector + col)*nNodesPerRow

    # Right face
    sideIdx = 3
    startRow = sideIdx * nSquaresPerRow
    endRow = (sideIdx + 1) * nSquaresPerRow
    offset = nNodesPerRow - 1

    for col in range(nNodesTriFace):
        Tb_Dirichlet[startRow:endRow, col] = (auxTbVector + col)*nNodesPerRow + offset


    return Tb_Dirichlet

def export_mesh(mesh_number, order, NODES, CONN, Tb_Dirichlet):
    # Name and directory of the mesh
    filename = f"mesh{mesh_number}_P{order}"
    directory = "mesh/"

    # Save
    np.savez_compressed(
            directory + filename,
            NODES = NODES,
            CONN = CONN,
            Tb_Dirichlet = Tb_Dirichlet
            )




if __name__ == "__main__":
    # Parameters
    mesh_numbers = range(0, 11)
    orders = range(1, 5)

    # Run the function
    for mesh_number in mesh_numbers:
        for order in orders:
            create_mesh(mesh_number, order)
