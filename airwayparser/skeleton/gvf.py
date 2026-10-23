import numpy as np
import tqdm
from numba import njit, prange

@njit(inline="always")
def _laplacian(a, i, j, k, hx, hy, hz):
    return ((a[i - 1, j, k] + a[i + 1, j, k] - 2 * a[i, j, k]) / hx**2
            + (a[i, j - 1, k] + a[i, j + 1, k] - 2 * a[i, j, k]) / hy**2
            + (a[i, j, k - 1] + a[i, j, k + 1] - 2 * a[i, j, k]) / hz**2)

@njit(parallel=True)
def _gvf_step(u, v, w, fx, fy, fz, mag_sq, mu, dt, hx, hy, hz, ii, jj, kk):
    for n in prange(len(ii)):
        i, j, k = ii[n], jj[n], kk[n]
        m = mag_sq[i, j, k]
        u[i, j, k] += dt * (mu * _laplacian(u, i, j, k, hx, hy, hz) - m * (u[i, j, k] - fx[i, j, k]))
        v[i, j, k] += dt * (mu * _laplacian(v, i, j, k, hx, hy, hz) - m * (v[i, j, k] - fy[i, j, k]))
        w[i, j, k] += dt * (mu * _laplacian(w, i, j, k, hx, hy, hz) - m * (w[i, j, k] - fz[i, j, k]))

def gvf(mask, spacing=(1.0, 1.0, 1.0), mu=0.15, dt=0.5, max_iters=500, verbose=True):
    f = np.pad(mask, 1).astype(float)
    hx, hy, hz = spacing
    if np.prod(spacing) / (6 * mu) < dt:  # keep the explicit scheme stable
        stable = np.prod(spacing) / (6 * mu) - 1e-4
        max_iters = int(max_iters * (dt / stable))
        dt = stable
        print(f"dt lowered to {dt:.4f} for stability, {max_iters} iterations")

    fx, fy, fz = np.gradient(f, hx, hy, hz)
    mag_sq = fx**2 + fy**2 + fz**2
    u, v, w = np.zeros_like(f), np.zeros_like(f), np.zeros_like(f)
    ii, jj, kk = np.nonzero(f)
    for _ in tqdm.trange(max_iters, disable=not verbose, desc="gvf"):
        _gvf_step(u, v, w, fx, fy, fz, mag_sq, mu, dt, hx, hy, hz, ii, jj, kk)
    return np.stack([u, v, w])[:, 1:-1, 1:-1, 1:-1]

def medial_function(mask, spacing=(1.0, 1.0, 1.0), mu=0.15, dt=0.5, max_iters=500, verbose=True):
    mask = np.asarray(mask, bool)
    field = gvf(mask, spacing, mu, dt, max_iters, verbose)
    norm = np.linalg.norm(field, axis=0)
    norm[norm == 0] = 1
    field /= norm
    div = sum(np.gradient(field[d], spacing[d], axis=d) for d in range(3))
    m = np.ma.MaskedArray(np.maximum(-div, 0.0), mask=~mask)
    return (m - m.min()) / (m.max() - m.min())
