"""Precision-independent diagnostics for the HDG Poisson paper runs.

The routines in this module are intentionally evaluated outside the measured
solver interval.  FP64 diagnostic arithmetic does not modify a computed FP32
or FP64 HDG solution; it only makes the reported norms insensitive to the
accumulation precision used by the performance implementation.
"""

from __future__ import annotations

import cupy as cp
import numpy as np


LAMBDA = 10.0
OMEGA = 6.0 * np.pi


def exact_flux_gpu(points_gpu, mu=1.0, dtype=cp.float64):
    """Return the manufactured flux q=-mu*grad(u) at physical points."""
    points_gpu = cp.asarray(points_gpu, dtype=dtype)
    x = points_gpu[..., 0]
    y = points_gpu[..., 1]
    lam = dtype(LAMBDA)
    omega = dtype(OMEGA)
    exp_y = cp.exp(-lam * y)
    cosine = cp.cos(omega * x)
    sine = cp.sin(omega * x)

    du_dx = 4.0 * lam**2 * omega * y * exp_y * sine
    du_dy = (
        8.0 * y
        - 4.0 * lam**2 * exp_y * (1.0 - lam * y) * cosine
        - 2.0 * lam**2 * cp.exp(-2.0 * lam * y)
    )
    return -dtype(mu) * cp.stack((du_dx, du_dy), axis=-1)


def source_gpu(points_gpu, mu=1.0, dtype=cp.float64):
    """Return f=-mu*Delta(u) for the non-polynomial manufactured solution."""
    points_gpu = cp.asarray(points_gpu, dtype=dtype)
    x = points_gpu[..., 0]
    y = points_gpu[..., 1]
    lam = dtype(LAMBDA)
    frequency = dtype(6.0)
    cosine = cp.cos(dtype(np.pi) * frequency * x)
    exp_y = cp.exp(-lam * y)
    return dtype(mu) * (
        4.0 * lam**4 * y * exp_y * cosine
        - 8.0 * lam**3 * exp_y * cosine
        - 4.0 * lam**3 * cp.exp(-2.0 * lam * y)
        - 4.0 * dtype(np.pi) ** 2 * frequency**2 * lam**2 * y * exp_y * cosine
        - 8.0
    )


def _volume_geometry(reference_element, nodes, connectivity, dtype):
    nodes_gpu = cp.asarray(nodes, dtype=dtype)
    connectivity_gpu = cp.asarray(connectivity, dtype=cp.int32)
    basis = cp.asarray(reference_element["N"], dtype=dtype)
    basis_xi = cp.asarray(reference_element["Nxi"], dtype=dtype)
    basis_eta = cp.asarray(reference_element["Neta"], dtype=dtype)
    weights = cp.asarray(reference_element["IPweights"], dtype=dtype)

    element_nodes = nodes_gpu[connectivity_gpu]
    points = cp.einsum("gn,enj->egj", basis, element_nodes)
    j11 = cp.einsum("gn,en->eg", basis_xi, element_nodes[:, :, 0])
    j12 = cp.einsum("gn,en->eg", basis_xi, element_nodes[:, :, 1])
    j21 = cp.einsum("gn,en->eg", basis_eta, element_nodes[:, :, 0])
    j22 = cp.einsum("gn,en->eg", basis_eta, element_nodes[:, :, 1])
    det_jacobian = j11 * j22 - j12 * j21
    differential_volume = cp.abs(det_jacobian) * weights[None, :]
    return basis, points, differential_volume


def compute_flux_l2_error(
    reference_element,
    nodes,
    connectivity,
    flux_values,
    mu=1.0,
    dtype=cp.float64,
):
    """Compute ||q_h-q||_L2 using FP64 diagnostic integration by default."""
    basis, points, differential_volume = _volume_geometry(
        reference_element, nodes, connectivity, dtype
    )
    n_elements = int(connectivity.shape[0])
    n_element_nodes = int(connectivity.shape[1])
    flux_gpu = cp.asarray(flux_values, dtype=dtype).reshape(
        n_elements, n_element_nodes, 2
    )
    flux_at_gauss = cp.einsum("gn,end->egd", basis, flux_gpu)
    exact_flux = exact_flux_gpu(points, mu=mu, dtype=dtype)
    error_squared = cp.sum((flux_at_gauss - exact_flux) ** 2, axis=2)
    return float(cp.sqrt(cp.sum(error_squared * differential_volume)).get())


def compute_local_conservation(
    elemental_trace_matrix,
    elemental_condensed_rhs,
    trace_solution,
    element_faces,
    reference_element,
    nodes,
    connectivity,
    mu=1.0,
    dtype=cp.float64,
):
    """Evaluate elementwise balance of numerical flux and volume source.

    The local condensed flux moments are ``Ke @ lambda_K - fe``.  Because the
    nodal face basis is a partition of unity, summing these moments over every
    face degree of freedom gives the integrated outward numerical flux for the
    element.  The returned relative norms use global source-integral scales,
    rather than division by each element's source integral.
    """
    faces_gpu = cp.asarray(element_faces, dtype=cp.int32)
    n_face_nodes = int(reference_element["N1d"].shape[1])
    trace_by_face = cp.asarray(trace_solution, dtype=dtype).reshape(
        -1, n_face_nodes
    )
    local_trace = trace_by_face[faces_gpu].reshape(faces_gpu.shape[0], -1)
    ke_gpu = cp.asarray(elemental_trace_matrix, dtype=dtype)
    fe_gpu = cp.asarray(elemental_condensed_rhs, dtype=dtype)
    flux_moments = cp.einsum("eij,ej->ei", ke_gpu, local_trace) - fe_gpu
    boundary_flux = cp.sum(flux_moments, axis=1)

    _, points, differential_volume = _volume_geometry(
        reference_element, nodes, connectivity, dtype
    )
    source_integral = cp.sum(
        source_gpu(points, mu=mu, dtype=dtype) * differential_volume,
        axis=1,
    )
    balance = boundary_flux - source_integral

    source_l2 = cp.linalg.norm(source_integral)
    source_linf = cp.max(cp.abs(source_integral))
    tiny = dtype(np.finfo(np.float64).tiny)
    relative_l2 = cp.linalg.norm(balance) / cp.maximum(source_l2, tiny)
    relative_linf = cp.max(cp.abs(balance)) / cp.maximum(source_linf, tiny)
    return {
        "conservation_relative_l2": float(relative_l2.get()),
        "conservation_relative_linf": float(relative_linf.get()),
        "conservation_max_abs": float(cp.max(cp.abs(balance)).get()),
        "conservation_source_l2": float(source_l2.get()),
    }
