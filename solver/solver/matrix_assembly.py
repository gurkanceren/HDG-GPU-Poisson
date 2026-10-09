
import time
import os

import cupy as cp
import numpy as np
from numba import cuda

from .numerical_integration import volumetric_integration_kernel, face_integration_kernel
# from .numerical_integration_component import volumetric_integration_kernel, face_integration_kernel
from .face_flipping import face_flipping_kernel
from .global_system import assemble_K_csr_disordered 
from .K_matrix import assemble_K_csr


def matrix_assembly(
        Nodes, Conn, Faces,
        referenceElement, infoFaces, 
        nComponents,
        mu, tau
        ):
    """
    This function assembles the elemental stiffness matrices from the mesh.
    Then, it converts into the desired global system format (i.e. CSR, DBF)
    """

    # --- Initialize data ---
    tic_0 = time.perf_counter()

    # Types for the data
    floatDtype = cp.float32
    intDtype = cp.int32

    # Important numbers
    nDim            = 2     # Number of dimensions of the problem
    nEl             = Conn.shape[0]
    nNodesPerEl     = referenceElement['N'].shape[1]
    nNodesPerFace   = referenceElement['NodesCoord1d'].shape[0]
    nIntFaces       = infoFaces['intFaces'].shape[0]
    nFacesPerEl     = 3     # For a triangle
    nBoundaryNodes  = nFacesPerEl * nNodesPerFace
    nTotFaces       = int(np.max(Faces)) + 1
    nGlobalDofs     = nTotFaces *nNodesPerFace 
    nGauss          = referenceElement['IPweights'].shape[0]    # Number of Gauss Nodes
    nGauss1d        = referenceElement['IPweights1d'].shape[0]    # Number of Gauss Nodes in 1D


    # --- Move data to GPU ---
    tic_1 = time.perf_counter()

    # Faces, nodes and element matrices
    NodesEl_gpu  = cp.asarray(Nodes[Conn],           dtype=floatDtype)
    Faces_gpu    = cp.asarray(Faces,                 dtype=intDtype)
    intFaces_gpu = cp.asarray(infoFaces['intFaces'], dtype=intDtype)
    extFaces_gpu = cp.asarray(infoFaces['extFaces'], dtype=intDtype)
    tau_gpu      = cp.asarray(tau,                   dtype=floatDtype)

    # Reference element matrices
    refN      = cp.asarray(referenceElement['N'],          dtype=floatDtype)
    refNxi    = cp.asarray(referenceElement['Nxi'],        dtype=floatDtype)
    refNeta   = cp.asarray(referenceElement['Neta'],       dtype=floatDtype)
    refIPw    = cp.asarray(referenceElement['IPweights'],  dtype=floatDtype)
    refN1d    = cp.asarray(referenceElement['N1d'],        dtype=floatDtype)
    refN1dxi  = cp.asarray(referenceElement['N1dxi'],      dtype=floatDtype)
    refIPw1d  = cp.asarray(referenceElement['IPweights1d'],dtype=floatDtype)
    faceNodes = cp.asarray(referenceElement['faceNodes'],  dtype=intDtype)


    # --- Initialize data in the GPU ---
    tic_2 = time.perf_counter()
    
    # Local Interior Matrices
    Auu = cp.zeros((nEl, nNodesPerEl,      nNodesPerEl),      dtype=floatDtype)
    Auq = cp.zeros((nEl, nNodesPerEl,      nDim*nNodesPerEl), dtype=floatDtype)
    Aqq = cp.zeros((nEl, nDim*nNodesPerEl, nDim*nNodesPerEl), dtype=floatDtype)

    # Face Matrices
    Alu = cp.zeros((nEl, nBoundaryNodes, nNodesPerEl),      dtype=floatDtype)
    Alq = cp.zeros((nEl, nBoundaryNodes, nDim*nNodesPerEl), dtype=floatDtype)
    All = cp.zeros((nEl, nBoundaryNodes, nBoundaryNodes),   dtype=floatDtype)

    # Right hand side
    fe = cp.zeros((nEl, nNodesPerEl),   dtype=floatDtype)

    
    # --- Volumetric integration ---
    tic_3 = time.perf_counter()

    # At least 1 thread per Gauss node, but more threads to avoid atomic operations
    threads_per_block = max(nGauss, nNodesPerEl*nNodesPerEl)    # Need to fulfill both constraints for the kernel
    blocks_per_grid = nEl   # 1 block per element

    # Debugging
    # nGauss = refIPw.shape[0]
    # print(f"nGauss number: {nGauss}\n\n")
    # nNodesPerEl = refN.shape[1]
    # print(f"NodesPerEl number: {nNodesPerEl}\n\n")
    # nGauss1d = refIPw1d.shape[0]
    # print(f"nGauss1D number: {nGauss1d}\n\n")

    volumetric_integration_kernel[blocks_per_grid, threads_per_block](
        NodesEl_gpu, refN, refNxi, refNeta, refIPw, mu,     # Inputs
        # nComponents,
        Aqq, Auq, fe                                        # Outputs
    )
    #TODO: Do I need this synchronize?? 
    # (I need for timings, but not sure if needed for later)
    cuda.synchronize()


    # --- Face integration ---
    tic_4 = time.perf_counter()

    boundary_1d_gauss_points = nFacesPerEl * nGauss1d
    matrix_entries = nFacesPerEl * nNodesPerFace * nNodesPerFace

    threads_per_block = max(boundary_1d_gauss_points, matrix_entries)   # Need to fulfill both constraints for the kernel
    blocks_per_grid = nEl   # 1 block per element

    face_integration_kernel[blocks_per_grid, threads_per_block](
        NodesEl_gpu, refN1d, refN1dxi, refIPw1d, faceNodes, tau_gpu, # Inputs
        Alq, Alu, All, Auu                                           # Outputs
    )
    cuda.synchronize()

    
    # --- Assemble and solve local systems ---
    tic_5 = time.perf_counter()

    # Obtain the rest of the matrices
    Aqu = -mu * Auq.transpose(0,2,1)
    Aul = -Alu.transpose(0,2,1)
    Aql = mu * Alq.transpose(0,2,1)

    # Join the matrices
    top = cp.concatenate([Auu, Auq], axis=2)
    bot = cp.concatenate([Aqu, Aqq], axis=2)
    Abig = cp.concatenate([top, bot], axis=1)

    # Right hand side matrices (Follow PhD Thesis formulation)
    # TODO: Can this be changed?
    rhsU = cp.concatenate([Aul, Aql],axis=1)    # [A_(u,uhat); A_(q,uhat)]
    tmp  = cp.zeros((nEl,2*nNodesPerEl),dtype=floatDtype)
    rhsf = cp.concatenate([fe[...,None],tmp[...,None]],axis=1).reshape((nEl,nNodesPerEl+2*nNodesPerEl, 1)) # [f_u; 0]

    # print(f"The shape of A is: {Abig.shape}")
    # print(f"The shape of rhsU is: {rhsU.shape}")
    # print(f"The shape of rhsf is: {rhsf.shape}")
    UQ = -cp.linalg.solve(Abig, rhsU)   # [U; Q]
    # fU =  cp.linalg.solve(Abig, rhsf)   # [f_U; f_Q]
    fU =  cp.linalg.solve(Abig, rhsf).squeeze()   # [f_U; f_Q]

    snapshot_path = os.environ.get("HDG_LOCAL_SNAPSHOT")
    snapshot = None
    if snapshot_path:
        sample_indices = np.unique(np.array(
            [0, nEl // 3, (2 * nEl) // 3, nEl - 1], dtype=np.int64
        ))
        snapshot = {
            "sample_indices": sample_indices,
            "Auu": cp.asnumpy(Auu[sample_indices]),
            "Auq": cp.asnumpy(Auq[sample_indices]),
            "Aqq": cp.asnumpy(Aqq[sample_indices]),
            "Alu": cp.asnumpy(Alu[sample_indices]),
            "Alq": cp.asnumpy(Alq[sample_indices]),
            "All": cp.asnumpy(All[sample_indices]),
            "source_fe": cp.asnumpy(fe[sample_indices]),
            "Abig": cp.asnumpy(Abig[sample_indices]),
            "UQ": cp.asnumpy(UQ[sample_indices]),
            "fU": cp.asnumpy(fU[sample_indices]),
        }
    # print(f"The shape of fU is: {fU.shape}")
    # print()


    # --- Extract local solutions ---
    tic_6 = time.perf_counter()

    dimU    = nNodesPerEl + 2*nNodesPerEl
    UQ_big  = UQ.reshape((nEl, dimU, nBoundaryNodes))
    fU_big  = fU.reshape((nEl, dimU))

    UU = UQ_big[:, :nNodesPerEl, :].copy()
    QQ = UQ_big[:, nNodesPerEl:, :].copy()
    Uf = fU_big[:, :nNodesPerEl].copy()
    Qf = fU_big[:, nNodesPerEl:].copy()
    QQ_nonflip = QQ.copy()
    # print(f"This is the non-flipped QQ Matrix (New): \n {QQ[1, ...]} \n\n")
    # print(f"This is the non-flipped All Matrix (New): \n {All[1, ...]} \n\n")


    # --- Flip local solutions where needed ---
    tic_7 = time.perf_counter()

    # Reshape matrices for easier data manipulation
    UU = UU.reshape(nEl, nNodesPerEl, nFacesPerEl, nNodesPerFace)
    QQ = QQ.reshape(nEl, nDim*nNodesPerEl, nFacesPerEl, nNodesPerFace)
    Alq = Alq.reshape(nEl, nFacesPerEl, nNodesPerFace, nDim*nNodesPerEl)
    Alu = Alu.reshape(nEl, nFacesPerEl, nNodesPerFace, nNodesPerEl)
    All = All.reshape(nEl, nFacesPerEl, nNodesPerFace, nFacesPerEl, nNodesPerFace)
    # print(f"This is the view of the UU Matrix (New): \n {UU[1, :, 2, :]} \n\n")
    # print(f"This is the view of the All Matrix (New): \n {All[1, 2, :, 2, :]} \n\n")


    # Parameters of the kernel
    threads_per_block = nFacesPerEl 
    blocks_per_grid = nEl 

    face_flipping_kernel[blocks_per_grid, threads_per_block](
            UU, QQ, Alq, Alu, All,      # Matrices to be flipped
            intFaces_gpu, Faces_gpu,
            nIntFaces, nFacesPerEl, nEl, nNodesPerEl, nDim, nNodesPerFace
            )
    cuda.synchronize()

    # Reshape back to normal size
    UU = UU.reshape(nEl, -1, nNodesPerFace * nFacesPerEl)
    QQ = QQ.reshape(nEl, -1, nNodesPerFace * nFacesPerEl)
    Alq = Alq.reshape(nEl, nNodesPerFace * nFacesPerEl, -1)
    Alu = Alu.reshape(nEl, nNodesPerFace * nFacesPerEl, -1)
    All = All.reshape(nEl, nNodesPerFace * nFacesPerEl, nNodesPerFace * nFacesPerEl)
    # print(f"This is the flipped QQ Matrix (New): \n {QQ[1, ...]} \n\n")
    # print(f"This is the flipped All Matrix (New): \n {All[1, ...]} \n\n")
    # print(f"This is the difference QQ Matrix (New): \n {QQ_nonflip[1, ...] - QQ[1, ...]} \n\n")


    # --- Build the elemental stiffness matrices ---
    tic_8 = time.perf_counter()

    Ke = Alq @ QQ + Alu @ UU + All
    fe = -(Alq @ Qf[..., cp.newaxis] + Alu @ Uf[..., cp.newaxis]).squeeze()
    if snapshot is not None:
        snapshot["Ke"] = cp.asnumpy(Ke[sample_indices])
        snapshot["condensed_fe"] = cp.asnumpy(fe[sample_indices])
        snapshot["UU"] = cp.asnumpy(UU[sample_indices])
        snapshot["QQ"] = cp.asnumpy(QQ[sample_indices])
        snapshot["Uf"] = cp.asnumpy(Uf[sample_indices])
        snapshot["Qf"] = cp.asnumpy(Qf[sample_indices])
        np.savez(snapshot_path, **snapshot)
        print(f"Saved local snapshot: {snapshot_path}")
    # print(f"\nThis is the Ke matrix: \n{Ke}\n")
    # print(f"\nThis is the fe vector: \n{fe}\n")

    # KeInv = cp.linalg.inv(Ke)



    #TODO: Create the function to assemble the global matrix
    tic_9 = time.perf_counter()

    K, f = assemble_K_csr(
                       Ke, fe,
                       Faces_gpu,
                       intFaces_gpu, extFaces_gpu,
                       nNodesPerFace, nFacesPerEl,
                       intDtype, floatDtype
                       )

    tic_10 = time.perf_counter()



    # Print the times of each function
    tic_11 = time.perf_counter()
    total_time = tic_11 - tic_0
    percent = lambda t: t / total_time * 100

    # print(f"\nEvaluating the Matrix Assembly")
    # print(f"\t \t Part 1: Initielize elements CPU")
    # print(f"\t \t Taken up {percent(tic_1 - tic_0):.1f}% ({tic_1 - tic_0:.3e}s)")
    # print(f"\t \t Part 2: Move to GPU")
    # print(f"\t \t Taken up {percent(tic_2 - tic_1):.1f}% ({tic_2 - tic_1:.3e}s)")
    # print(f"\t \t Part 3: Initialize buffers GPU")
    # print(f"\t \t Taken up {percent(tic_3 - tic_2):.1f}% ({tic_3 - tic_2:.3e}s)")
    # print(f"\t \t Part 4: Volumetric integration")
    # print(f"\t \t Taken up {percent(tic_4 - tic_3):.1f}% ({tic_4 - tic_3:.3e}s)")
    # print(f"\t \t Part 5: Face integration")
    # print(f"\t \t Taken up {percent(tic_5 - tic_4):.1f}% ({tic_5 - tic_4:.3e}s)")
    # print(f"\t \t Part 6: Assemble and solve local systems")
    # print(f"\t \t Taken up {percent(tic_6 - tic_5):.1f}% ({tic_6 - tic_5:.3e}s)")
    # print(f"\t \t Part 7: Extract local solutions")
    # print(f"\t \t Taken up {percent(tic_7 - tic_6):.1f}% ({tic_7 - tic_6:.3e}s)")
    # print(f"\t \t Part 8: Flip local solutions")
    # print(f"\t \t Taken up {percent(tic_8 - tic_7):.1f}% ({tic_8 - tic_7:.3e}s)")
    # print(f"\t \t Part 9: Create Ke and fe")
    # print(f"\t \t Taken up {percent(tic_9 - tic_8):.1f}% ({tic_9 - tic_8:.3e}s)")
    # print(f"\t \t Part 10: Assemble K in CSR format")
    # print(f"\t \t Taken up {percent(tic_10 - tic_9):.1f}% ({tic_10 - tic_9:.3e}s)")
    # print(f"\t \t Part 11: Assemble the f vector")
    # print(f"\t \t Taken up {percent(tic_11 - tic_10):.1f}% ({tic_11 - tic_10:.3e}s)")
    # print(f"\n \t \t Function took {total_time:.4f}s\n \n")


    return K, Ke, f, fe, QQ, UU, Qf, Uf
