import numpy as np

def load_mesh(mesh_number, order):
    # Create filename
    filename = f"mesh/mesh{mesh_number}_P{order}.npz"

    # Open file
    mesh = np.load(filename)

    # Extract data
    NODES = mesh["NODES"]
    CONN = mesh["CONN"]
    Tb_Dirichlet = mesh["Tb_Dirichlet"]

    # Close file
    mesh.close()

    return NODES, CONN, Tb_Dirichlet
