import time

# kernel_HDG_postprocess_vectorized_3.py
import numpy as np
import cupy as cp
from numba import cuda, float64

from referenceElement.evaluateNodalBasisTriWithoutDerivatives import (
    evaluate_nodal_basis_tri_without_derivatives
)

# -------------------------------------------------------------------------------
# CUDA kernel: assemble local matrices for HDG postprocess
# -------------------------------------------------------------------------------

@cuda.jit
def hdg_postprocess_kernel(
    Xe,    # (nElem, nLoc, 2)
    N,     # (ngauss, npts)
    Nxi,   # (ngauss, npts)
    Neta,  # (ngauss, npts)
    dNdxi, # (ngauss, nLoc)
    dNdeta,# (ngauss, nLoc)
    IPw,   # (ngauss,)
    ue,    # (nElem, npts)         # already in enriched nodes
    qx,    # (nElem, npts)
    qy,    # (nElem, npts)
    K_out, # (nElem, npts, npts)   # OUTPUT
    Bq_out,# (nElem, npts)         # OUTPUT
    Ius_out,#(nElem, npts)         # OUTPUT
    Iu_out # (nElem,)              # OUTPUT
):
    e = cuda.blockIdx.x
    t = cuda.threadIdx.x

    ngauss = IPw.shape[0]
    npts   = N.shape[1]
    nLoc   = Xe.shape[1]

    # Acummulators to avoid race conditions
    acc_Iu = 0.0

    # gauss loop
    for g in range(ngauss):
        # Jacobian
        J11 = 0.0; J12 = 0.0; J21 = 0.0; J22 = 0.0
        for a in range(nLoc):
            x = Xe[e, a, 0]
            y = Xe[e, a, 1]
            J11 += dNdxi[g, a]  * x
            J12 += dNdxi[g, a]  * y
            J21 += dNdeta[g, a] * x
            J22 += dNdeta[g, a] * y
        det = J11 * J22 - J12 * J21
        invJ11 =  J22 / det; invJ12 = -J12 / det
        invJ21 = -J21 / det; invJ22 =  J11 / det
        dvol   = IPw[g] * det

        # u_g, qx_g, qy_g (in star nodes)
        u_g = 0.0; qx_g = 0.0; qy_g = 0.0
        for a in range(npts):
            Na = N[g, a]
            u_g  += Na * ue[e, a]
            qx_g += Na * qx[e, a]
            qy_g += Na * qy[e, a]

        # ∫u dΩ lo accumulates only once
        if t == 0:
            acc_Iu += u_g * dvol

        # for each shape function Ni
        if t < npts:
            # grad Ni
            Nx = invJ11 * Nxi[g, t] + invJ12 * Neta[g, t]
            Ny = invJ21 * Nxi[g, t] + invJ22 * Neta[g, t]

            # Bq[i] += (∇Ni · q) dΩ
            val_bq = (Nx * qx_g + Ny * qy_g) * dvol
            cuda.atomic.add(Bq_out, (e, t), val_bq)

            # Ius[i] += Ni dΩ
            val_ius = N[g, t] * dvol
            cuda.atomic.add(Ius_out, (e, t), val_ius)

            # row of K[i,j] += (∇Ni·∇Nj) dΩ
            for j in range(npts):
                Nxj = invJ11 * Nxi[g, j] + invJ12 * Neta[g, j]
                Nyj = invJ21 * Nxi[g, j] + invJ22 * Neta[g, j]
                kij = (Nx * Nxj + Ny * Nyj) * dvol
                cuda.atomic.add(K_out, (e, t, j), kij)

    # write∫u dΩ
    if t == 0:
        Iu_out[e] = acc_Iu



def HDG_postprocess_gpu(X, T, u_flat, q_flat, ref_star):
    """
    GPU‑vectorized HDG post‑process con CuPy,
    recibe u_flat (nElem*nLoc,) y q_flat (nElem*nLoc,2).
    """

    # Start timer of this function
    tic_0 = time.perf_counter()

    nElem, nLoc = T.shape
    npts        = ref_star['NodesCoord'].shape[0]
    deg         = ref_star['degree'] - 1


    # CPU-side: calculate N_star (npts, nLoc)
    tic_1 = time.perf_counter()

    N_star_cpu = evaluate_nodal_basis_tri_without_derivatives(
        ref_star['NodesCoord'],
        ref_star['NodesCoordGeo'],
        deg
    )  # (npts, nLoc)

    # Extract data to CPU
    tic_2 = time.perf_counter()
    Xe_cpu  = X[T]                            # (nElem, nLoc, 2)
    u_e_cpu = u_flat.reshape(nElem, nLoc)     # (nElem, nLoc)
    q_e_cpu = q_flat.reshape(nElem, nLoc, 2)  # (nElem, nLoc, 2)

    # Projection to enriched space
    tic_3 = time.perf_counter()
    ue_star_cpu = u_e_cpu @ N_star_cpu.T          # (nElem, npts)
    qx_star_cpu = q_e_cpu[:,:,0] @ N_star_cpu.T   # (nElem, npts)
    qy_star_cpu = q_e_cpu[:,:,1] @ N_star_cpu.T   # (nElem, npts)

    # Move data to GPU
    tic_4 = time.perf_counter()
    Xe   = cp.asarray(Xe_cpu)
    ue   = cp.asarray(ue_star_cpu)
    qx   = cp.asarray(qx_star_cpu)
    qy   = cp.asarray(qy_star_cpu)
    N     = cp.asarray(ref_star['N'])
    Nxi   = cp.asarray(ref_star['Nxi'])
    Neta  = cp.asarray(ref_star['Neta'])
    dNdxi = cp.asarray(ref_star['dNGeodxi'])
    dNdeta= cp.asarray(ref_star['dNGeodeta'])
    IPw   = cp.asarray(ref_star['IPweights'])

    # Prepare output arrays
    tic_5 = time.perf_counter()
    K_out   = cp.zeros((nElem, npts, npts), dtype=cp.float64)
    Bq_out  = cp.zeros((nElem, npts)     , dtype=cp.float64)
    Ius_out = cp.zeros((nElem, npts)     , dtype=cp.float64)
    Iu_out  = cp.zeros((nElem, )         , dtype=cp.float64)

    # Lauch kernel
    tic_6 = time.perf_counter()
    threads = max(npts, IPw.size)
    print(f"""The number of threads is the maximum between: 
                  - npts: {npts}
                  - IPw.size: {IPw.size} """)   # IPw.size = 54 for P4
    hdg_postprocess_kernel[nElem, threads](
        Xe, N, Nxi, Neta, dNdxi, dNdeta, IPw,
        ue, qx, qy,
        K_out, Bq_out, Ius_out, Iu_out
    )
    cuda.synchronize()

    # Assemble and solve local systems on CPU
    tic_7 = time.perf_counter()
    A   = cp.zeros((nElem, npts+1, npts+1), dtype=cp.float64)
    RHS = cp.zeros((nElem, npts+1)       , dtype=cp.float64)

    A[:nElem, :npts, :npts] = K_out[:nElem]
    A[:nElem, :npts,  npts] = Ius_out[:nElem]
    A[:nElem, npts, :npts]  = Ius_out[:nElem]
    RHS[:nElem, :npts]      = Bq_out[:nElem]
    RHS[:nElem, npts]       = Iu_out[:nElem]

    sol    = cp.linalg.solve(A, RHS)
    u_star = sol[:, :npts].ravel()

    tic_8 = time.perf_counter()

    # Print the times of each function
    total_time = tic_8 - tic_0
    percent = lambda t: t / total_time * 100

    # print(f"\nEvaluating the Post-Processing")
    # print(f"\t \t Part 1: Initialize data")
    # print(f"\t \t Taken up {percent(tic_1 - tic_0):.1f}% ({tic_1 - tic_0:.3e}s)")
    # print(f"\t \t Part 2: Calculate N_star CPU")
    # print(f"\t \t Taken up {percent(tic_2 - tic_1):.1f}% ({tic_2 - tic_1:.3e}s)")
    # print(f"\t \t Part 3: Extract data to CPU")
    # print(f"\t \t Taken up {percent(tic_3 - tic_2):.1f}% ({tic_3 - tic_2:.3e}s)")
    # print(f"\t \t Part 4: Projection to the enriched space")
    # print(f"\t \t Taken up {percent(tic_4 - tic_3):.1f}% ({tic_4 - tic_3:.3e}s)")
    # print(f"\t \t Part 5: Move data to GPU")
    # print(f"\t \t Taken up {percent(tic_5 - tic_4):.1f}% ({tic_5 - tic_4:.3e}s)")
    # print(f"\t \t Part 6: Prepare output arrays")
    # print(f"\t \t Taken up {percent(tic_6 - tic_5):.1f}% ({tic_6 - tic_5:.3e}s)")
    # print(f"\t \t Part 7: Launch post-process kernel")
    # print(f"\t \t Taken up {percent(tic_7 - tic_6):.1f}% ({tic_7 - tic_6:.3e}s)")
    # print(f"\t \t Part 8: Assemble and solve local systems on CPU")
    # print(f"\t \t Taken up {percent(tic_8 - tic_7):.1f}% ({tic_8 - tic_7:.3e}s)")
    # print(f"\n \t \t Function took {total_time:.4f}s\n \n")
    #
    return u_star
