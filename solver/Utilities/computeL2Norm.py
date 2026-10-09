# Utilities/computeL2NormGPU.py
import time
import cupy as cp
import numpy as np

# if _accumulate_ms is defined in __main__, use it to accumulate CPU-GPU transfer time
try:
    from __main__ import _accumulate_ms
except Exception:
    def _accumulate_ms(_):  # no-op
        pass

def compute_L2_norm(reference_element, X_dev, T_dev, u_dev, u0,
                        u0_is_gpu=False, dtype=cp.float32):
    """
    Compute the L2 norm of the error between a discrete function u_dev
    defined on a triangular mesh and an analytical function u0. The mesh
    is given by the coordinates X_dev and the connectivity T_dev.       
    """
    # to GPU
    X_dev = cp.asarray(X_dev)
    T_dev = cp.asarray(T_dev)
    u_dev = cp.asarray(u_dev).reshape(T_dev.shape[0], T_dev.shape[1]).astype(dtype, copy=False)

    N    = cp.asarray(reference_element['N']).astype(dtype, copy=False)
    Nxi  = cp.asarray(reference_element['Nxi']).astype(dtype, copy=False)
    Neta = cp.asarray(reference_element['Neta']).astype(dtype, copy=False)
    IPw  = cp.asarray(reference_element['IPweights']).astype(dtype, copy=False)

    #  batched ASSEMBLE
    Xe   = X_dev[T_dev]                                     # (Ne,nEn,2)
    xy_g = cp.einsum('gn,enj->egj', N, Xe)                  # (Ne,ng,2)
    ue_g = cp.einsum('gn,en->eg',   N, u_dev)               # (Ne,ng)

    J11 = cp.einsum('gn,en->eg', Nxi,  Xe[:,:,0])
    J12 = cp.einsum('gn,en->eg', Nxi,  Xe[:,:,1])
    J21 = cp.einsum('gn,en->eg', Neta, Xe[:,:,0])
    J22 = cp.einsum('gn,en->eg', Neta, Xe[:,:,1])
    detJ = J11*J22 - J12*J21
    dvol = detJ * IPw[None,:]

    # u0 IN GPU or CPU and compute error
    if u0_is_gpu:
        u0_g = u0(xy_g.reshape(-1,2)).reshape(ue_g.shape)
    else:
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        u0_g = cp.asarray(u0(cp.asnumpy(xy_g).reshape(-1,2))).reshape(ue_g.shape)
        cp.cuda.Stream.null.synchronize()
        _accumulate_ms(time.perf_counter() - t0)  # ACCUMULATES     CPU

    err2 = (ue_g - u0_g)**2 * dvol
    return float(cp.sqrt(cp.sum(err2)).get())
