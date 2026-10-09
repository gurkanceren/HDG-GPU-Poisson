import numpy as np

def orthogonal_polynomials_and_derivatives_1D(degree, Xi):
    """
    Computes orthogonal polynomials and their derivatives for 1D reference interval.
    
    Parameters:
    degree (int): Polynomial degree.
    Xi (ndarray): Coordinates (N,1) where the polynomials are evaluated.
    
    Returns:
    P (ndarray): Evaluated orthogonal polynomials.
    dPdxi (ndarray): Derivatives w.r.t xi.
    """
    "x = Xi[:, 0]"
    if Xi.ndim == 1:
        x = Xi  # Si es un vector 1D, úsalo directamente
    else:
        x = Xi[:, 0]  # Si es 2D, extrae la primera columna

    U = np.ones_like(x)
    O = np.zeros_like(x)
    
    if degree == 1:
        P = np.vstack([U, x]).T
        dPdxi = np.vstack([O, U]).T
    elif degree == 2:
        P = np.vstack([U, x, -U/2 + (3/2) * x**2]).T
        dPdxi = np.vstack([O, U, 3 * x]).T
    elif degree == 3:
        P = np.vstack([U, x, -U/2 + (3/2) * x**2, (5/2) * x**3 - (3/2) * x]).T
        dPdxi = np.vstack([O, U, 3 * x, (15/2) * x**2 - 3/2 * U]).T
    elif degree == 4:
        P = np.vstack([U, x, -U/2 + (3/2) * x**2, (5/2) * x**3 - (3/2) * x, 3/8 * U + (35/8) * x**4 - (15/4) * x**2]).T
        dPdxi = np.vstack([O, U, 3 * x, (15/2) * x**2 - 3/2 * U, (35/2) * x**3 - (15/2) * x]).T
    elif degree == 5:
        P = np.vstack([U, x, -U/2 + (3/2) * x**2, (5/2) * x**3 - (3/2) * x, 3/8 * U + (35/8) * x**4 - (15/4) * x**2, (63/8) * x**5 - (35/4) * x**3 + (15/8) * x]).T
        dPdxi = np.vstack([O, U, 3 * x, (15/2) * x**2 - 3/2 * U, (35/2) * x**3 - (15/2) * x, (315/8) * x**4 - (105/4) * x**2 + 15/8]).T
    else:
        raise ValueError("Degree not implemented")
    
    return P, dPdxi