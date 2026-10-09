import math

from numba import cuda, float32


# -------------------------------------------------------------------------
# Source term
# -------------------------------------------------------------------------
@cuda.jit(device=True)
def source_poisson_kernel(x, y, mu):
    lam = float32(10.0)
    frequency = float32(6.0)
    cosine = math.cos(frequency * math.pi * x)
    exp_y = math.exp(-lam * y)
    return mu * (
        4.0 * lam**4 * y * exp_y * cosine
        - 8.0 * lam**3 * exp_y * cosine
        - 4.0 * lam**3 * math.exp(-2.0 * lam * y)
        - 4.0 * math.pi**2 * frequency**2 * lam**2 * y * exp_y * cosine
        - 8.0
    )


# -------------------------------------------------------------------------
# Volumetric Integration
# -------------------------------------------------------------------------
@cuda.jit
def volumetric_integration_kernel(
    NodesEl, refN, refNxi, refNeta, refIPw, mu,  # Inputs
    Aqq, Auq, fe                                     # Outputs
):
    """
    Kernel that computes the volumetric integrals Aqq, Auq.
    Important note: This kernel uses a hybrid approach to avoid atomic operations.
        The first (initialization) part of the code uses a thread for each Gauss Node. 
        The second (computation and writing) part uses a thread for every entry of the local matrix Aqq (nNodesPerEl x nNodesPerEl)
    """
    floatDtype = float32

    # Threads identification
    e = cuda.blockIdx.x
    tx = cuda.threadIdx.x
    
    nNodesPerEl = refN.shape[1]
    nGauss = refIPw.shape[0]

    # Maximum number of Gauss points (for P4)
    # Needed to know shared memory size at compilation time
    nGaussMax = 54
    nNodesPerElMax = 15
    
    # Create shared arrays for fast memory
    sh_dvol = cuda.shared.array(nGaussMax, floatDtype)
    sh_src = cuda.shared.array(nGaussMax, floatDtype)
    sh_Nx = cuda.shared.array((nGaussMax, nNodesPerElMax), floatDtype) 
    sh_Ny = cuda.shared.array((nGaussMax, nNodesPerElMax), floatDtype)

    # Geometry Precomputation (Threads map to Gauss points)
    if tx < nGauss:
        # More meaningful name
        gaussPoint = tx

        # Initialize values of Jacobian and Gauss Node position
        J11 = J12 = J21 = J22 = 0.0
        xg = yg = 0.0
        
        # Loop over nodes to obtain Jacobian and Physical coordinates of Gauss Nodes
        for a in range(nNodesPerEl):
            xi = refNxi[gaussPoint, a]
            eta = refNeta[gaussPoint, a]
            x0 = NodesEl[e, a, 0]
            y0 = NodesEl[e, a, 1]
            Nv = refN[gaussPoint, a]
            
            J11 += xi * x0;    J12 += xi * y0
            J21 += eta * x0;   J22 += eta * y0
            xg += Nv * x0;     yg += Nv * y0
            
        detJ = J11*J22 - J12*J21
        invJ11 =  J22 / detJ; invJ12 = -J12 / detJ
        invJ21 = -J21 / detJ; invJ22 =  J11 / detJ
        
        # Cache the scaled integration weight and source term
        sh_dvol[gaussPoint] = refIPw[gaussPoint] * detJ
        sh_src[gaussPoint] = source_poisson_kernel(xg, yg, mu)
        
        # Cache physical gradients
        for a in range(nNodesPerEl):
            sh_Nx[gaussPoint, a] = invJ11*refNxi[gaussPoint,a] + invJ12*refNeta[gaussPoint,a]
            sh_Ny[gaussPoint, a] = invJ21*refNxi[gaussPoint,a] + invJ22*refNeta[gaussPoint,a]

    cuda.syncthreads() # Ensure all Gauss point data is visible to all threads

    # Matrix Assembly (Threads map to i, j pairs)
    nEntries = nNodesPerEl * nNodesPerEl
    
    if tx < nEntries:
        i = tx // nNodesPerEl
        j = tx % nNodesPerEl
        
        # Hardware registers for accumulation (no atomic operations)
        reg_M = 0.0
        reg_Auq_x = 0.0
        reg_Auq_y = 0.0
        
        # Single loop over Gauss points
        for q in range(nGauss):
            dvol = sh_dvol[q]
            Ni = refN[q, i]
            Nj = refN[q, j]
            
            # Mass matrix component Aqq
            reg_M += Ni * Nj * dvol
            
            # Divergence matrix component Auq
            reg_Auq_x += Ni * sh_Nx[q, j] * dvol
            reg_Auq_y += Ni * sh_Ny[q, j] * dvol

        # Aqq is block diagonal, mass matrix repeated for x and y components
        Aqq[e, (2*i), (2*j)] = reg_M
        Aqq[e, (2*i + 1), (2*j + 1)] = reg_M
        
        # Auq is NOT block diagonal, need x and y components
        Auq[e, i, (2*j)] = reg_Auq_x
        Auq[e, i, (2*j + 1)] = reg_Auq_y

    # Vector f is 1D (in the element space)
    # Performing the loop separate to the main assembly avoids thread divergence
    if tx < nNodesPerEl:
        reg_f = 0.0
        for q in range(nGauss):
            reg_f += refN[q, tx] * sh_src[q] * sh_dvol[q]
            
        fe[e, tx] = reg_f




# -------------------------------------------------------------------------
# Face Integration
# -------------------------------------------------------------------------
@cuda.jit
def face_integration_kernel(
    NodesEl, refN1d, refN1dxi, refIPw1d, faceNodes, tauElem, # Inputs
    Alq, Alu, All, Auu                                       # Outputs
):
    """
    Kernel that computes the face integrals Alq, Alu, All, Auu.
    Follows the hybrid approach to avoid atomics where possible.

        The first part maps threads to Gauss Nodes across all 3 faces.
        The second part maps threads to entries of the local trace matrices.
    """
    floatDtype = float32

    # Threads identification
    e = cuda.blockIdx.x
    tx = cuda.threadIdx.x
    
    nFacesPerEl = 3     # Triangles
    nNodesPerFace = faceNodes.shape[1]
    nGauss1d = refIPw1d.shape[0]
    nNodesPerEl = NodesEl.shape[1]
    nBoundaryNodes = nFacesPerEl * nNodesPerFace

    # Maximum number of Gauss points (for P4)
    # Needed to know shared memory size at compilation time
    nGauss1DMax = 9
    
    # Create shared arrays for fast memory (sized for nFacesPerEl faces)
    sh_dline = cuda.shared.array((nFacesPerEl, nGauss1DMax), floatDtype)
    sh_Nx = cuda.shared.array((nFacesPerEl, nGauss1DMax), floatDtype)
    sh_Ny = cuda.shared.array((nFacesPerEl, nGauss1DMax), floatDtype)

    # Geometry Precomputation (Threads map to Gauss points across nFacesPerEl faces)
    total_gauss_threads = nFacesPerEl * nGauss1d
    
    if tx < total_gauss_threads:
        iface = tx // nGauss1d
        q = tx % nGauss1d
        
        dxdxi = 0.0
        dydxi = 0.0
        
        # Loop over face nodes to obtain 1D Jacobian and physical coordinates
        for a in range(nNodesPerFace):
            idx = faceNodes[iface, a]
            dN = refN1dxi[q, a]
            dxdxi += dN * NodesEl[e, idx, 0]
            dydxi += dN * NodesEl[e, idx, 1]
            
        norm = math.hypot(dxdxi, dydxi)
        
        if norm > 1e-6:
            sh_Nx[iface, q] = dydxi / norm
            sh_Ny[iface, q] = -dxdxi / norm
            # Cache scaled integration weight (norm represents the differential line element)
            sh_dline[iface, q] = norm * refIPw1d[q]
        else:
            sh_Nx[iface, q] = 0.0
            sh_Ny[iface, q] = 0.0
            sh_dline[iface, q] = 0.0

    cuda.syncthreads() # Ensure all 1D Gauss point data is visible

    # Matrix Assembly (Threads map to i, j face node pairs)
    nFaceEntries = nNodesPerFace * nNodesPerFace
    total_matrix_threads = nFacesPerEl * nFaceEntries
    
    if tx < total_matrix_threads:
        iface = tx // nFaceEntries
        remainder = tx % nFaceEntries
        i_1d = remainder // nNodesPerFace
        j_1d = remainder % nNodesPerFace
        
        tau_f = tauElem[e, iface]
        
        # Hardware registers for accumulation (no atomic operations)
        reg_Alq_x = 0.0
        reg_Alq_y = 0.0
        reg_Alu = 0.0
        
        # Single loop over 1D Gauss points
        for q in range(nGauss1d):
            dline = sh_dline[iface, q]
            Nx = sh_Nx[iface, q]
            Ny = sh_Ny[iface, q]
            
            Ni = refN1d[q, i_1d]
            Nj = refN1d[q, j_1d]
            
            Ni_Nj_dline = Ni * Nj * dline
            
            reg_Alq_x += Ni_Nj_dline * Nx
            reg_Alq_y += Ni_Nj_dline * Ny
            reg_Alu += Ni_Nj_dline * tau_f
            
        # Global and local indices
        i_loc = iface * nNodesPerFace + i_1d
        j_loc = iface * nNodesPerFace + j_1d
        
        node_i = faceNodes[iface, i_1d]
        node_j = faceNodes[iface, j_1d]
        
        # Write directly to global N-dimensional arrays
        Alq[e, i_loc, (2*node_j)] = reg_Alq_x
        Alq[e, i_loc, (2*node_j + 1)] = reg_Alq_y
        
        Alu[e, i_loc, node_j] = reg_Alu
        All[e, i_loc, j_loc] = -reg_Alu
        
        # Auu strictly requires atomic add because corner nodes are shared by adjacent faces
        cuda.atomic.add(Auu, (e, node_i, node_j), reg_Alu)
