import numpy as np
from numba import njit
from scipy.ndimage import distance_transform_edt


@njit(cache=True, inline="always")
def _axis(n, x):
    i = int(np.floor(x))
    if i < 0:
        return 0, 0, 0.0
    if i >= n - 1:
        return n - 1, n - 1, 0.0
    return i, i + 1, x - i


@njit(cache=True, inline="always")
def _interp(T, x, y, z):
    i0, i1, tx = _axis(T.shape[0], x)
    j0, j1, ty = _axis(T.shape[1], y)
    k0, k1, tz = _axis(T.shape[2], z)
    c00 = T[i0, j0, k0] * (1 - tz) + T[i0, j0, k1] * tz
    c01 = T[i0, j1, k0] * (1 - tz) + T[i0, j1, k1] * tz
    c10 = T[i1, j0, k0] * (1 - tz) + T[i1, j0, k1] * tz
    c11 = T[i1, j1, k0] * (1 - tz) + T[i1, j1, k1] * tz
    return (c00 * (1 - ty) + c01 * ty) * (1 - tx) + (c10 * (1 - ty) + c11 * ty) * tx


@njit(cache=True, inline="always")
def _inside(mask, x, y, z):
    n0, n1, n2 = mask.shape
    if x < 0 or y < 0 or z < 0 or x > n0 - 1 or y > n1 - 1 or z > n2 - 1:
        return False
    return mask[int(np.floor(x + 0.5)), int(np.floor(y + 0.5)), int(np.floor(z + 0.5))]


@njit(cache=True, inline="always")
def _grad_dir(T, sp, x, y, z):
    # unit vector down the gradient of T, in voxel index units; ok is False where the gradient vanishes
    gx = (_interp(T, x + 0.5, y, z) - _interp(T, x - 0.5, y, z)) / sp[0]
    gy = (_interp(T, x, y + 0.5, z) - _interp(T, x, y - 0.5, z)) / sp[1]
    gz = (_interp(T, x, y, z + 0.5) - _interp(T, x, y, z - 0.5)) / sp[2]
    dx, dy, dz = -gx / sp[0], -gy / sp[1], -gz / sp[2]
    n = np.sqrt(dx * dx + dy * dy + dz * dz)
    if n == 0.0 or not np.isfinite(n):
        return False, 0.0, 0.0, 0.0
    return True, dx / n, dy / n, dz / n


@njit(cache=True)
def _snap(T, mask, i, j, k):
    n0, n1, n2 = T.shape
    best = T[i, j, k]
    bi, bj, bk = -1, -1, -1
    for a in range(max(i - 1, 0), min(i + 2, n0)):
        for b in range(max(j - 1, 0), min(j + 2, n1)):
            for c in range(max(k - 1, 0), min(k + 2, n2)):
                if mask[a, b, c] and T[a, b, c] < best:
                    best = T[a, b, c]
                    bi, bj, bk = a, b, c
    return bi, bj, bk


@njit(cache=True)
def _descend(T, mask, sp, source, start, h, path):
    px, py, pz = start[0], start[1], start[2]
    hstop = min(sp[0], min(sp[1], sp[2]))
    n = 0
    while n < len(path) - 1:
        path[n, 0], path[n, 1], path[n, 2] = px, py, pz
        n += 1
        ex, ey, ez = (px - source[0]) * sp[0], (py - source[1]) * sp[1], (pz - source[2]) * sp[2]
        d = np.sqrt(ex * ex + ey * ey + ez * ez)
        if d <= hstop:
            if d > 0.0:
                path[n, 0], path[n, 1], path[n, 2] = source[0], source[1], source[2]
                n += 1
            return n

        qx, qy, qz = px, py, pz
        ok, k1x, k1y, k1z = _grad_dir(T, sp, px, py, pz)
        if ok:
            ok, k2x, k2y, k2z = _grad_dir(T, sp, px + 0.5 * h * k1x, py + 0.5 * h * k1y, pz + 0.5 * h * k1z)
            if ok:
                qx, qy, qz = px + h * k2x, py + h * k2y, pz + h * k2z
                ok = _inside(mask, qx, qy, qz) and _interp(T, qx, qy, qz) < _interp(T, px, py, pz)
        if not ok:
            i, j, k = _snap(T, mask, int(np.floor(px + 0.5)), int(np.floor(py + 0.5)), int(np.floor(pz + 0.5)))
            if i < 0:
                return n  # nothing lower around: this is the source
            qx, qy, qz = 1.0 * i, 1.0 * j, 1.0 * k
        px, py, pz = qx, qy, qz
    return n


class Tracer:
    def __init__(self, travel_time, mask, root, spacing=(1.0, 1.0, 1.0), h=0.5, max_steps=20000):
        self.mask = np.ascontiguousarray(mask, bool)
        T = np.ma.filled(travel_time, np.inf).astype(float)
        nearest = distance_transform_edt(~self.mask, return_distances=False, return_indices=True)
        self.T = T[tuple(nearest)]  # voxels outside copy their nearest inside voxel, so the gradient exists at the wall
        self.root = np.asarray(root, float)
        self.spacing = np.asarray(spacing, float)
        self.h = h
        self.path = np.empty((max_steps, 3))

    def trace(self, endpoint):
        n = _descend(self.T, self.mask, self.spacing, self.root, np.asarray(endpoint, float), self.h, self.path)
        return self.path[:n][::-1].copy()
