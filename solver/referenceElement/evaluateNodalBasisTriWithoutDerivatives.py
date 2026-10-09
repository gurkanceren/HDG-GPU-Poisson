import numpy as np

from referenceElement.orthogonalPolynomialsTri import orthogonal_polynomials_tri
from referenceElement.orthogonalPolynomialsAndDerivativesTri import orthogonal_polynomials_and_derivatives_tri

def evaluate_nodal_basis_tri_without_derivatives(XI, XInodes, degree):
    """
    Evaluates the nodal basis functions at points XI for a triangular element,
    given the interpolation nodes XInodes and the degree of the polynomial.
    This function does NOT return derivatives.
    
    Parameters:
    - XI: Evaluation points (nPoints x 2)
    - XInodes: Interpolation nodes (nNodes x 2)
    - degree: Polynomial degree (int)

    Returns:
    - N: Nodal basis functions evaluated at XI (nPoints x nNodes)
    """
    # Compute the Vandermonde matrix at interpolation nodes
    V = orthogonal_polynomials_tri(degree, XInodes)  # shape (nNodes x nBasis)
    invV = np.linalg.inv(V)  # shape (nBasis x nNodes)

    # Evaluate orthogonal basis at XI
    P, _, _ = orthogonal_polynomials_and_derivatives_tri(degree, XI)  # P: (nPoints x nBasis)

    # Change of basis from orthogonal to nodal
    N = np.dot(P, invV)  # shape (nPoints x nNodes)

    return N
