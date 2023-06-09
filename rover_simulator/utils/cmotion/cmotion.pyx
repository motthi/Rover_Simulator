import cython
import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, acos, exp, sqrt, fabs, erf, M_PI
from cython import boundscheck, wraparound

ctypedef np.float64_t FLOAT64_T

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:] state_transition(
    double[:] pose,
    FLOAT64_T nu,
    FLOAT64_T omega,
    FLOAT64_T dt
):
    cdef FLOAT64_T t0 = pose[2]
    cdef double[:] new_pose
    new_pose = np.zeros(3)
    if omega < 1e-10 and omega > -1e-10:
        new_pose[0] = pose[0] + nu * cos(t0) * dt
        new_pose[1] = pose[1] + nu * sin(t0) * dt
        new_pose[2] = pose[2] + omega * dt
    else:
        new_pose[0] = pose[0] + nu / omega * (sin(t0 + omega * dt) - sin(t0))
        new_pose[1] = pose[1] + nu / omega * (-cos(t0 + omega * dt) + cos(t0))
        new_pose[2] = pose[2] + omega * dt
    while new_pose[2] > M_PI:
        new_pose[2] -= 2 * M_PI
    while new_pose[2] < -M_PI:
        new_pose[2] += 2 * M_PI
    return new_pose

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:, :] covariance_transition(
    double[:] pose,
    double[:, :] cov,
    double[:] stds,
    FLOAT64_T nu,
    FLOAT64_T omega,
    FLOAT64_T dt
):
    cdef FLOAT64_T theta = pose[2]
    cdef FLOAT64_T st = sin(theta)
    cdef FLOAT64_T ct = cos(theta)
    cdef FLOAT64_T stw = sin(theta + omega * dt)
    cdef FLOAT64_T ctw = cos(theta + omega * dt)
    cdef FLOAT64_T nu_per_omega = nu / omega
    cdef FLOAT64_T nu_per_omega2 = nu / (omega**2)
    cdef FLOAT64_T absnu_per_dt = abs(nu) / dt
    cdef FLOAT64_T absomega_per_dt = abs(omega) / dt
    cdef FLOAT64_T A11, A12, A21, A22, A32, M11, M22, F13, F23
    cdef double[:, :] new_cov
    
    A11 = (stw - st) / omega
    A12 = -nu_per_omega2 * (stw - st) + nu_per_omega * dt * ctw
    A21 = (-ctw + ct) / omega
    A22 = -nu_per_omega2 * (-ctw + ct) + nu_per_omega * dt * stw
    A32 = dt
    
    M11 = stds[0]**2 * absnu_per_dt + stds[1]**2 * absomega_per_dt
    M22 = stds[2]**2 * absnu_per_dt + stds[3]**2 * absomega_per_dt
    
    F13 = nu_per_omega * (cos(theta + omega * dt) - cos(theta))
    F23 = nu_per_omega * (sin(theta + omega * dt) - sin(theta))
    
    new_cov = np.zeros((3, 3))
    new_cov[0, 0] = cov[0, 0] + cov[0, 2] * F13 + F13 * cov[2, 0] + F13 * F13 * cov[2, 2] + A11 * A11 * M11 + A12 * A12 * M22
    new_cov[0, 1] = cov[0, 1] + cov[0, 2] * F23 + F13 * cov[2, 1] + F13 * cov[2, 2] * F23 + A11 * A21 * M11 + A12 * A22 * M22
    new_cov[0, 2] = cov[0, 2] + F13 * cov[2, 2] + A12 * A32 * M22
    new_cov[1, 0] = cov[1, 0] + cov[1, 2] * F13 + F23 * cov[2, 0] + cov[2, 2] * F13 * F23 + A21 * A11 * M11 + A22 * M22 * A12
    new_cov[1, 1] = cov[1, 1] + cov[1, 2] * F23 + F23 * cov[2, 1] + F23 * F23 * cov[2, 2] + A21 * A21 * M11 + M22 * A22 * A22
    new_cov[1, 2] = cov[1, 2] + F23 * cov[2, 2] + A32 * A22 * M22
    new_cov[2, 0] = cov[2, 0] + cov[2, 2] * F13 + A12 * A32 * M22
    new_cov[2, 1] = cov[2, 1] + cov[2, 2] * F23 + A22 * A32 * M22
    new_cov[2, 2] = cov[2, 2] + A32 * A32 * M22
    return new_cov

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef FLOAT64_T prob_collision(
    double[:] x,
    double[:, :] cov,
    double[:] obs_pos,
    FLOAT64_T obs_r,
):
    cdef FLOAT64_T a_ij0
    cdef FLOAT64_T a_ij1
    cdef FLOAT64_T b_ij = obs_r
    cdef FLOAT64_T dist
    cdef FLOAT64_T delta0
    cdef FLOAT64_T a_cov
    cdef FLOAT64_T delta1
    cdef FLOAT64_T prob_col
    dist = sqrt((x[0] - obs_pos[0])**2 + (x[1] - obs_pos[1])**2)
    delta0 = x[0] - obs_pos[0]
    delta1 = x[1] - obs_pos[1]
    a_ij0 = delta0 / dist
    a_ij1 = delta1 / dist
    a_cov = a_ij0 * (cov[0, 0] * a_ij0 + cov[0, 1] * a_ij1) + a_ij1 * (cov[1, 0] * a_ij0 + cov[1, 1] * a_ij1)
    prob_col = (1.0 + erf((b_ij - (a_ij0 * delta0 + a_ij1 * delta1)) / sqrt(2.0 * a_cov))) / 2
    return prob_col