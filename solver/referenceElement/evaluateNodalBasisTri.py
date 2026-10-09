import numpy as np

from referenceElement.orthogonalPolynomialsTri import orthogonal_polynomials_tri
from referenceElement.orthogonalPolynomialsAndDerivativesTri import orthogonal_polynomials_and_derivatives_tri

def evaluate_nodal_basis_tri(XI, XI_nodes, degree):
    """
    Evaluates nodal basis functions and their derivatives on a triangular reference element.

    Parameters
    ----------
    XI : ndarray
        Points where the basis is evaluated, shape (nPoints, 2).
    XI_nodes : ndarray
        Coordinates of the interpolation nodes in the reference triangle, shape (nNodes, 2).
    degree : int
        Polynomial degree of the basis.

    Returns
    -------
    N : ndarray
        Shape functions evaluated at XI, shape (nPoints, nNodes).
    dN_dxi : ndarray
        Derivative w.r.t. xi, shape (nPoints, nNodes).
    dN_deta : ndarray
        Derivative w.r.t. eta, shape (nPoints, nNodes).
    """

    # Compute Vandermonde matrix at interpolation nodes
    V = orthogonal_polynomials_tri(degree, XI_nodes)  # shape (nNodes, nBasis)
    invV = np.linalg.inv(V)

    # Compute orthogonal polynomials and their derivatives at evaluation points
    P, dP_dxi, dP_deta = orthogonal_polynomials_and_derivatives_tri(degree, XI)  # all shape (nPoints, nBasis)

    # Change of basis to nodal basis
    N = P @ invV
    dN_dxi = dP_dxi @ invV
    dN_deta = dP_deta @ invV

    return N, dN_dxi, dN_deta
