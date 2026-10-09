import time

# kernel_HDG_postprocess_vectorized_3.py
import numpy as np
import cupy as cp
from numba import cuda, float64

from referenceElement.evaluateNodalBasisTriWithoutDerivatives import (
    evaluate_nodal_basis_tri_without_derivatives
)



def HDG_postprocess_gpu(X, T, u_flat, q_flat, ref_star):
    """
    Receives u_flat (nEl*nNodesPerEl,) y q_flat (nEl*nNodesPerEl,2).

    GPU adapted HDG post-process kernel that computes the volumetric integrals Asq, Ks.
    The "s" subscript notates superconverged solution "u_star".
    Important note: This kernel uses a hybrid approach to avoid atomic operations.
        The first (initialization) part of the code uses a thread for each Gauss Node. 
        The second (computation and writing) part uses a thread for every entry of the local matrix Aqq (nNodesPerEl x nNodesPerEl)

    """

    # Start timer of this function
    tic_0 = time.perf_counter()

    nEl, nNodesPerEl = T.shape
    npts        = ref_star['NodesCoord'].shape[0]
    deg         = ref_star['degree'] - 1


    # CPU-side: calculate N_star (npts, nNodesPerEl)
    tic_1 = time.perf_counter()

    N_star_cpu = evaluate_nodal_basis_tri_without_derivatives(
        ref_star['NodesCoord'],
        ref_star['NodesCoordGeo'],
        deg
    )  # (npts, nNodesPerEl)

    # Extract data to CPU
    tic_2 = time.perf_counter()
    NodesEl_cpu  = X[T]                            # (nEl, nNodesPerEl, 2)
    u_e_cpu = u_flat.reshape(nEl, nNodesPerEl)     # (nEl, nNodesPerEl)
    q_e_cpu = q_flat.reshape(nEl, nNodesPerEl, 2)  # (nEl, nNodesPerEl, 2)

    # Projection to enriched space
    tic_3 = time.perf_counter()
    ue_star_cpu = u_e_cpu @ N_star_cpu.T          # (nEl, npts)
    qx_star_cpu = q_e_cpu[:,:,0] @ N_star_cpu.T   # (nEl, npts)
    qy_star_cpu = q_e_cpu[:,:,1] @ N_star_cpu.T   # (nEl, npts)

    # Move data to GPU
    tic_4 = time.perf_counter()
    NodesEl   = cp.asarray(NodesEl_cpu)
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
    K_out   = cp.zeros((nEl, npts, npts), dtype=cp.float64)
    Bq_out  = cp.zeros((nEl, npts)     , dtype=cp.float64)
    Ius_out = cp.zeros((nEl, npts)     , dtype=cp.float64)
    Iu_out  = cp.zeros((nEl, )         , dtype=cp.float64)

    # Lauch kernel
    tic_6 = time.perf_counter()
    threads_per_block = max(npts*npts, IPw.size)
    blocks_per_grid = nEl
    hdg_postprocess_kernel[blocks_per_grid, threads_per_block](
        NodesEl, N, Nxi, Neta, dNdxi, dNdeta, IPw,
        ue, qx, qy,
        K_out, Bq_out, Ius_out, Iu_out
    )
    cuda.synchronize()

    # Assemble and solve local systems on CPU
    tic_7 = time.perf_counter()
    A   = cp.zeros((nEl, npts+1, npts+1), dtype=cp.float64)
    RHS = cp.zeros((nEl, npts+1)       , dtype=cp.float64)

    A[:nEl, :npts, :npts] = K_out[:nEl]
    A[:nEl, :npts,  npts] = Ius_out[:nEl]
    A[:nEl, npts, :npts]  = Ius_out[:nEl]
    RHS[:nEl, :npts]      = Bq_out[:nEl]
    RHS[:nEl, npts]       = Iu_out[:nEl]

    #sol    = cp.linalg.solve(A, RHS)
    #u_star = sol[:, :npts].ravel()
     
    # Add a final dimension so RHS is a batch of column vectors
    sol = cp.linalg.solve(A, RHS[..., None])
    # Remove that final singleton dimension
    sol = sol[..., 0]

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



@cuda.jit
def hdg_postprocess_kernel(
    NodesEl,             # Coordinates of the nodes
    N, Nxi, Neta, dNdxi, dNdeta, IPw,   # Reference element functions
    u, qx, qy,      # Elemental solutions
    Ks, Bq, uStarInt, uInt  # Outputs
    ):

    """
    GPU adapted HDG post-process kernel that computes the volumetric integrals Ks, Bq, subjected 
    to the mean value constraint.
    The "s" subscript notates superconverged solution "u_star".
    Important note: This kernel uses a hybrid approach to avoid atomic operations.
        The first (initialization) part of the code uses a thread for each Gauss Node. 
        The second (computation and writing) part uses a thread for every entry of the local matrix 
            - This part is divided in the computation of matrix, vector and scalar.
            - These are separated to avoid warp divergence and maximize performance (Hopefully)
    """
    floatDtype = cp.float32

    # Thread identification
    el = cuda.blockIdx.x
    tx = cuda.threadIdx.x

    nGauss = IPw.shape[0]
    nNodesPerEl   = NodesEl.shape[1]
    nScPoints   = N.shape[1]    # Number of superconverged points 
                                # (Number of points there would be with degree + 1)

    # Maximum number of variables
    # Needed to know shared memory size at compilation time
    nGaussMax = 54
    nScPointsMax = 21

    # Create shared arrays for fast memory
    sh_dvol = cuda.shared.array(nGaussMax, floatDtype)
    sh_NGauss = cuda.shared.array((nGaussMax, nScPointsMax), floatDtype)
    sh_Nx = cuda.shared.array((nGaussMax, nScPointsMax), floatDtype)
    sh_Ny = cuda.shared.array((nGaussMax, nScPointsMax), floatDtype)

    sh_u = cuda.shared.array(nGaussMax, floatDtype)
    sh_qx = cuda.shared.array(nGaussMax, floatDtype)
    sh_qy = cuda.shared.array(nGaussMax, floatDtype)


    # Geometry precomputation (threads map to Gauss points)
    if tx < nGauss:
        # More meaningful name
        gaussPoint = tx

        # Initialize values of Jacobian and Gauss Node Positions
        J11 = 0.0; J12 = 0.0; J21 = 0.0; J22 = 0.0

        # Loop over nodes to obtain Jacobian and Physical coordinates of Gauss Nodes
        for node in range(nNodesPerEl):
            x = NodesEl[el, node, 0]
            y = NodesEl[el, node, 1]

            J11 += dNdxi[gaussPoint, node]  * x
            J12 += dNdxi[gaussPoint, node]  * y
            J21 += dNdeta[gaussPoint, node] * x
            J22 += dNdeta[gaussPoint, node] * y

        # Determinant and inverse values
        detJ = J11 * J22 - J12 * J21
        invJ11 =  J22 / detJ;   invJ12 = -J12 / detJ
        invJ21 = -J21 / detJ;   invJ22 =  J11 / detJ

        # Cache integration weight
        sh_dvol[gaussPoint] = IPw[gaussPoint] * detJ

        # Obtain the physical gradients
        for scPoint in range(nScPoints):
            sh_Nx[gaussPoint, scPoint] = invJ11 * Nxi[gaussPoint, scPoint] + invJ12 * Neta[gaussPoint, scPoint] 
            sh_Ny[gaussPoint, scPoint] = invJ21 * Nxi[gaussPoint, scPoint] + invJ22 * Neta[gaussPoint, scPoint] 

        # Obtain the elemental solutions
        uGauss = 0.0; qxGauss = 0.0; qyGauss = 0.0;
        for scPoint in range(nScPoints):
            NGauss = N[gaussPoint, scPoint]
            sh_NGauss[gaussPoint, scPoint] = NGauss

            uGauss  += NGauss *  u[el, scPoint]
            qxGauss += NGauss * qx[el, scPoint]
            qyGauss += NGauss * qy[el, scPoint]

        sh_u[gaussPoint] = uGauss
        sh_qx[gaussPoint] = qxGauss
        sh_qy[gaussPoint] = qyGauss

    cuda.syncthreads() # Ensure all Gauss point data is visible to all threads

    # Second part. Matrix Assembly (Threads map to i, j pairs)
    nEntries = nScPoints * nScPoints

    if tx < nEntries:
        i = tx // nScPoints
        j = tx % nScPoints

        # Use registers for accumulation
        reg_Ks = 0.0

        # Single loop over Gauss Points
        for gaussPoint in range(nGauss):
            # Obtain data from shared memory
            dvol = sh_dvol[gaussPoint]
            Nx_i = sh_Nx[gaussPoint, i]
            Nx_j = sh_Nx[gaussPoint, j]
            Ny_i = sh_Ny[gaussPoint, i]
            Ny_j = sh_Ny[gaussPoint, j]

            reg_Ks += (Nx_i*Nx_j + Ny_i*Ny_j) * dvol

        Ks[el, i, j] = reg_Ks


    # Third part. Vector Assembly (For the Bq and uStarInt)
    if tx < nScPoints:
        # More meaningful name
        scPoint = tx

        # Use registers for accumulation
        reg_Bq = 0.0
        reg_uStar = 0.0

        # Single loop over Gauss Points
        for gaussPoint in range(nGauss):
            # Obtain data from shared memory
            dvol = sh_dvol[gaussPoint]
            NGauss = sh_NGauss[gaussPoint, scPoint]
            Nx = sh_Nx[gaussPoint, scPoint]
            Ny = sh_Ny[gaussPoint, scPoint]
            qx = sh_qx[gaussPoint]
            qy = sh_qy[gaussPoint]

            reg_Bq += (Nx*qx + Ny*qy) * dvol
            reg_uStar += NGauss * dvol

        Bq[el, scPoint] = reg_Bq
        uStarInt[el, scPoint] = reg_uStar


    # Fourth part. Scalar Assembly (For the uInt)
    if tx == 0:

        # Use registers for accumulation
        reg_u = 0.0

        # Single loop over Gauss Points
        for gaussPoint in range(nGauss):
            # Obtain data from shared memory
            dvol = sh_dvol[gaussPoint]
            u = sh_u[gaussPoint]

            reg_u += u * dvol

        uInt[el] = reg_u


