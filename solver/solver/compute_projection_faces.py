# hdgCommon/computeProjectionFaces_gpu.py

import numpy as np
import cupy as cp
import math
from numba import cuda, float32, int32

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

    return u.astype(np.float32)

@cuda.jit
def project_faces_elem_kernel(
    Xe,             # (nElems, nNodesPerElem, 2) float32
    extFaceIdx,     # (nElems, 3) int32, -1 if interior face
    faceNodes,      # (3, nFaceNodes) int32
    N1d,            # (ngf, nFaceNodes) float32
    N1dxi,          # (ngf, nFaceNodes) float32
    IPw1d,          # (ngf,) float32
    ug_global,      # (nExtFaces, ngf) float32
    u_out,          # (nExtFaces * nFaceNodes,) float32
    nFaceNodes,
    ngf
):
    e = cuda.blockIdx.x
    t = cuda.threadIdx.x

    sm_M = cuda.shared.array((3, 32, 32), float32)
    sm_b = cuda.shared.array((3, 32),       float32)

    # intialize
    if t == 0:
        for f_loc in range(3):
            for i in range(nFaceNodes):
                sm_b[f_loc, i] = 0.0
                for j in range(nFaceNodes):
                    sm_M[f_loc, i, j] = 0.0
    cuda.syncthreads()

    # assemble
    if t < ngf:
        for f_loc in range(3):
            extId = extFaceIdx[e, f_loc]
            if extId < 0:
                continue

            xfg = 0.0; yfg = 0.0
            dxdxi = 0.0; dydxi = 0.0
            for a in range(nFaceNodes):
                Ni   = N1d[t, a]
                Nxi  = N1dxi[t, a]
                loc  = faceNodes[f_loc, a]
                x    = Xe[e, loc, 0]
                y    = Xe[e, loc, 1]
                xfg   += Ni * x
                yfg   += Ni * y
                dxdxi += Nxi * x
                dydxi += Nxi * y

            ug = ug_global[extId, t]
            w  = IPw1d[t] * math.hypot(dxdxi, dydxi)

            for i in range(nFaceNodes):
                Ni = N1d[t, i]
                for j in range(nFaceNodes):
                    Nj = N1d[t, j]
                    cuda.atomic.add(sm_M, (f_loc, i, j), Ni * Nj * w)
                cuda.atomic.add(sm_b, (f_loc, i), Ni * ug * w)
    cuda.syncthreads()

    # solve locally and write output
    if t == 0:
        for f_loc in range(3):
            extId = extFaceIdx[e, f_loc]
            if extId < 0:
                continue

            for k in range(nFaceNodes):
                inv = 1.0 / sm_M[f_loc, k, k]
                for j in range(k+1, nFaceNodes):
                    fac = sm_M[f_loc, j, k] * inv
                    for l in range(k, nFaceNodes):
                        sm_M[f_loc, j, l] -= fac * sm_M[f_loc, k, l]
                    sm_b[f_loc, j] -= fac * sm_b[f_loc, k]

            base = extId * nFaceNodes
            for i in range(nFaceNodes-1, -1, -1):
                acc = sm_b[f_loc, i]
                for j in range(i+1, nFaceNodes):
                    acc -= sm_M[f_loc, i, j] * sm_b[f_loc, j]
                sm_b[f_loc, i] = acc / sm_M[f_loc, i, i]
                u_out[base + i] = sm_b[f_loc, i]


def compute_projection_faces_gpu(faces, X, T, reference_element):
    
        # This setup section runs on the CPU
    if isinstance(faces, cp.ndarray):
        faces = cp.asnumpy(faces)

    if isinstance(X, cp.ndarray):
        X = cp.asnumpy(X)

    if isinstance(T, cp.ndarray):
        T = cp.asnumpy(T)

    faceNodes = reference_element["faceNodes"]
    if isinstance(faceNodes, cp.ndarray):
        faceNodes = cp.asnumpy(faceNodes)

    N1d = reference_element["N1d"]
    if isinstance(N1d, cp.ndarray):
        N1d = cp.asnumpy(N1d)

    IPw1d = reference_element["IPweights1d"]
    if isinstance(IPw1d, cp.ndarray):
        IPw1d = cp.asnumpy(IPw1d)
    
    faceNodes   = reference_element["faceNodes"]
    N1d         = reference_element["N1d"]
    N1dxi       = reference_element["N1dxi"]
    IPw1d       = reference_element["IPweights1d"]
    nFaceNodes  = faceNodes.shape[1]
    ngf         = IPw1d.shape[0]
    nElems      = T.shape[0]
    nExtFaces   = faces.shape[0]

    # precompute ug (CPU)
    ug_host = np.zeros((nExtFaces, ngf), dtype=np.float32)
    for idx in range(nExtFaces):
        e, f_loc = faces[idx]
        # Convert NumPy integer scalars to ordinary Python integers
        e = int(e)
        f_loc = int(f_loc)
        nodes    = faceNodes[f_loc]
        xf       = X[T[e]][nodes, 0]
        yf       = X[T[e]][nodes, 1]
        xfg      = N1d @ xf
        yfg      = N1d @ yf
        pts      = np.column_stack((xfg, yfg))
        ug_host[idx] = analytical_poisson(pts)
        
    # Outside the loop: one CPU-to-GPU transfer
    ug_dev = cp.asarray(ug_host, dtype=cp.float32)    

    extFaceIdx = -np.ones((nElems, 3), dtype=np.int32)
    for idx in range(nExtFaces):
        e, f_loc = faces[idx]
        extFaceIdx[e, f_loc] = idx

    Xe_dev         = cp.asarray(X[T],       dtype=cp.float32)
    extFaceIdx_dev = cp.asarray(extFaceIdx, dtype=cp.int32)
    faceNodes_dev  = cp.asarray(faceNodes,  dtype=cp.int32)
    N1d_dev        = cp.asarray(N1d,        dtype=cp.float32)
    N1dxi_dev      = cp.asarray(N1dxi,      dtype=cp.float32)
    IPw1d_dev      = cp.asarray(IPw1d,      dtype=cp.float32)
    ug_dev         = cp.asarray(ug_host,    dtype=cp.float32)
    u_dev          = cp.zeros(nExtFaces * nFaceNodes, dtype=cp.float32)

    threads = max(ngf, nFaceNodes)
    project_faces_elem_kernel[nElems, threads](
        Xe_dev,
        extFaceIdx_dev,
        faceNodes_dev,
        N1d_dev,
        N1dxi_dev,
        IPw1d_dev,
        ug_dev,
        u_dev,
        nFaceNodes,
        ngf
    )
    cuda.synchronize()

    return u_dev
