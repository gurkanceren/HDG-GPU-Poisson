
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.spatial import Delaunay
from referenceElement.evaluateNodalBasisTriWithoutDerivatives import (
    evaluate_nodal_basis_tri_without_derivatives
)

def plot_postprocessed_solution(X, T, u, reference_element_star, n_deg_ref=20):
    

   
    u = np.asarray(u)
    if u.ndim == 2:
        n_elements, n_nodes_per_elem = u.shape
        get_u_local = lambda ielem: u[ielem, :]
    else:
        coord_ref = reference_element_star["NodesCoord"]
        n_nodes_per_elem = coord_ref.shape[0]
        n_elements = T.shape[0]
        get_u_local = lambda ielem: u[ielem * n_nodes_per_elem : (ielem + 1) * n_nodes_per_elem]

  
    nodes_ref = []
    h = 1.0 / n_deg_ref
    for j in range(n_deg_ref + 1):
        for i in range(n_deg_ref + 1 - j):
            nodes_ref.append([i * h, j * h])
    nodes_ref = np.array(nodes_ref)
    nodes_ref = 2 * nodes_ref - 1 

   
    tri_ref = Delaunay(nodes_ref).simplices

    coord_ref = reference_element_star["NodesCoord"]
    coord_geo = reference_element_star["NodesCoordGeo"]
    degree    = reference_element_star["degree"]

    
    N     = evaluate_nodal_basis_tri_without_derivatives(nodes_ref, coord_ref, degree)
    N_geo = evaluate_nodal_basis_tri_without_derivatives(nodes_ref, coord_geo, degree - 1)

    x_all = []
    y_all = []
    u_all = []
    tri_global = []

    for ielem in range(n_elements):
        Xe = X[T[ielem, :], :]                
        u_local = get_u_local(ielem)           

        Xg = N_geo @ Xe                       
        ug = N @ u_local                       

        idx_start = len(x_all)
        x_all.extend(Xg[:, 0])
        y_all.extend(Xg[:, 1])
        u_all.extend(ug)
        tri_global.extend((tri_ref + idx_start).tolist())

    x_all = np.array(x_all)
    y_all = np.array(y_all)
    u_all = np.array(u_all)
    tri_global = np.array(tri_global)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    triang = Triangulation(x_all, y_all, tri_global)
    tpc = ax.tripcolor(triang, u_all, shading='gouraud', cmap='jet')
    ax.set_aspect('equal')
    ax.set_title("Postprocessed HDG solution: $u^*$")

    return fig, ax, tpc
