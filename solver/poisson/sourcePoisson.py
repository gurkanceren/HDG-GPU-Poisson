def sourcePoisson(X, mu):
    """
    Source function for the Poisson problem.
    
    Parameters:
      X : np.ndarray
          Array of points (n_points x 2).
      mu : float
          Coefficient for the current element.
    
    Returns:
      s : np.ndarray
          Source term evaluated at each point.
    """
    # Parameters
    lam = 10  # 'lambda' is a reserved word in Python
    f = 6
    
    # Extract coordinates
    x = X[:, 0]
    y = X[:, 1]
    
    # Consistent manufactured source s = -mu * Laplacian(u_exact), where
    # u_exact is defined in analyticalPoisson.py.
    s = mu * (
        4 * lam**4 * y * np.exp(-lam * y) * np.cos(np.pi * f * x)
        - 8 * lam**3 * np.exp(-lam * y) * np.cos(np.pi * f * x)
        - 4 * lam**3 * np.exp(-2 * lam * y)
        - 4 * np.pi**2 * f**2 * lam**2 * y * np.exp(-lam * y)
        * np.cos(np.pi * f * x)
        - 8
    )
    return s
