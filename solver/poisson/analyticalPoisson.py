import numpy as np

def analytical_poisson(X):
    """
    Returns the analytical solution u(x, y) for validation or Dirichlet BCs.
    
    Parameters
    ----------
    X : ndarray of shape (nPoints, 2)
        Array of (x, y) coordinates where the solution is evaluated.
    
    Returns
    -------
    u : ndarray
        The solution u(x, y) evaluated at each point.
    """
    lambda_ = 10
    f = 6

    x = X[:, 0]
    y = X[:, 1]

    # Smooth non-polynomial manufactured solution.
    u = (
        4*y**2
        - 4*lambda_**2 * y * np.exp(-lambda_ * y) * np.cos(f * np.pi * x)
        + lambda_ * np.exp(-2 * lambda_ * y)
    )

    return u
