# hdgCommon/computeElementsSolution_gpu.py

import cupy as cp
from numba import cuda, float32, int32

@cuda.jit
def compute_elem_solution_kernel(
    lambda_d,      # (nFaces * nFaceNodes,) float32
    UU_d,          # (nElems, nElemNodes, 3*nFaceNodes) float32
    QQ_d,          # (nElems, 2*nElemNodes, 3*nFaceNodes) float32
    Uf_d,          # (nElems, nElemNodes) float32
    Qf_d,          # (nElems, 2*nElemNodes) float32
    F_d,           # (nElems, 3) int32
    u_out,         # (nElems * nElemNodes,) float32
    q_out,         # (nElems * 2*nElemNodes,) float32
    nFaceNodes,
    nElemNodes
):
    e = cuda.blockIdx.x
    t = cuda.threadIdx.x

    # reconstruct u
    if t < nElemNodes:
        acc_u = 0.0
        for fi in range(3):
            face = F_d[e, fi]
            base = face * nFaceNodes
            for j in range(nFaceNodes):
                acc_u += UU_d[e, t, fi * nFaceNodes + j] * lambda_d[base + j]
        u_out[e * nElemNodes + t] = acc_u + Uf_d[e, t]

    # reconstruct q
    if t < 2 * nElemNodes:
        acc_q = 0.0
        for fi in range(3):
            face = F_d[e, fi]
            base = face * nFaceNodes
            for j in range(nFaceNodes):
                acc_q += QQ_d[e, t, fi * nFaceNodes + j] * lambda_d[base + j]
        q_out[e * 2 * nElemNodes + t] = acc_q + Qf_d[e, t]


def compute_elements_solution_gpu(lambda_d, UU_d, QQ_d, Uf_d, Qf_d, F_d):
    """
    GPU-only version: todas las entradas deben ser CuPy arrays ya en dispositivo.

    Parameters
    ----------
    lambda_d : cupy.ndarray, shape (nFaces * nFaceNodes,)
    UU_d      : cupy.ndarray, shape (nElems, nElemNodes, 3*nFaceNodes)
    QQ_d      : cupy.ndarray, shape (nElems, 2*nElemNodes, 3*nFaceNodes)
    Uf_d      : cupy.ndarray, shape (nElems, nElemNodes)
    Qf_d      : cupy.ndarray, shape (nElems, 2*nElemNodes)
    F_d       : cupy.ndarray (int32), shape (nElems, 3)

    Returns
    -------
    u : cupy.ndarray, shape (nElems * nElemNodes,)
    q : cupy.ndarray, shape (nElems * nElemNodes, 2)
    """
    nElems     = F_d.shape[0]
    nElemNodes = UU_d.shape[1]
    nFaceNodes = UU_d.shape[2] // 3

    # allocate output arrays
    u_d = cp.zeros(nElems * nElemNodes,       dtype=cp.float32)
    q_d = cp.zeros(nElems * 2 * nElemNodes,   dtype=cp.float32)

    # launch kernel
    threads_per_block = max(nElemNodes, 2 * nElemNodes)
    compute_elem_solution_kernel[
        nElems, threads_per_block
    ](
        lambda_d, UU_d, QQ_d, Uf_d, Qf_d, F_d,
        u_d, q_d,
        nFaceNodes, nElemNodes
    )
    cuda.synchronize()

    # reorder q_d to shape (nElems*nElemNodes, 2)
    q_d = q_d.reshape((nElems * nElemNodes, 2))

    return u_d, q_d
