# Utilities/computeL2NormPostProcessedGPU.py
import time
import cupy as cp
import numpy as np

try:
    from __main__ import _accumulate_ms
except Exception:
    def _accumulate_ms(_):
        pass

def compute_L2_norm_postprocess(reference_element_star, X_dev, T_dev,
                                    u_star_dev, u0, u0_is_gpu=False, dtype=cp.float32):
    """
    Compute the L2 norm of the error of the postprocessed solution. The
    postprocessed solution is assumed to be given at the element nodes. The
    exact solution u0 is a function that takes as input a (N,2) array       
    """
    X_dev = cp.asarray(X_dev)
    T_dev = cp.asarray(T_dev)
    u_star = cp.asarray(u_star_dev).reshape(T_dev.shape[0], -1).astype(dtype, copy=False)

    N      = cp.asarray(reference_element_star['N']).astype(dtype, copy=False)
    NGeo   = cp.asarray(reference_element_star['NGeo']).astype(dtype, copy=False)
    NxiGeo = cp.asarray(reference_element_star['dNGeodxi']).astype(dtype, copy=False)
    NetaGeo= cp.asarray(reference_element_star['dNGeodeta']).astype(dtype, copy=False)
    IPw    = cp.asarray(reference_element_star['IPweights']).astype(dtype, copy=False)

    Xe   = X_dev[T_dev]                                     # (Ne,nEn,2)
    xy_g = cp.einsum('gn,enj->egj', NGeo, Xe)               # (Ne,ng,2)
    ue_g = cp.einsum('gn,en->eg',   N,    u_star)           # (Ne,ng)

    J11 = cp.einsum('gn,en->eg', NxiGeo,  Xe[:,:,0])
    J12 = cp.einsum('gn,en->eg', NxiGeo,  Xe[:,:,1])
    J21 = cp.einsum('gn,en->eg', NetaGeo, Xe[:,:,0])
    J22 = cp.einsum('gn,en->eg', NetaGeo, Xe[:,:,1])
    detJ = J11*J22 - J12*J21
    dvol = detJ * IPw[None,:]

    if u0_is_gpu:
        u0_g = u0(xy_g.reshape(-1,2)).reshape(ue_g.shape)
    else:
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        u0_g = cp.asarray(u0(cp.asnumpy(xy_g).reshape(-1,2))).reshape(ue_g.shape)
        cp.cuda.Stream.null.synchronize()
        _accumulate_ms(time.perf_counter() - t0)

    err2 = (ue_g - u0_g)**2 * dvol
    return float(cp.sqrt(cp.sum(err2)).get())
