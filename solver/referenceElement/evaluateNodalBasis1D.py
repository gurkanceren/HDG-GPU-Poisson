import numpy as np

"""import importlib
importlib.reload(referenceElement.evaluateNodalBasis1D)"""

from referenceElement.orthogonalPolynomialsAndDerivatives1D import orthogonal_polynomials_and_derivatives_1D

def evaluate_nodal_basis_1D(Xi, XiNodes, degree):
    """
    Evaluates nodal basis of polynomials for a given degree in 1D.
    """
    V, _ = orthogonal_polynomials_and_derivatives_1D(degree, XiNodes)
    P, dPdxi = orthogonal_polynomials_and_derivatives_1D(degree, Xi)
    
    N = np.linalg.solve(V.T, P.T).T  # Equivalent to P/V in MATLAB
    dNdxi = np.linalg.solve(V.T, dPdxi.T).T  # Equivalent to dPdxi/V in MATLAB
    
    return N, dNdxi