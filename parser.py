import numpy as np
import skfmm
from scipy.ndimage import distance_transform_edt, label

from gvf import medial_function
from tracer import Tracer
from tree import BranchTree


def parse_tree(mask, spacing=(1.0, 1.0, 1.0), mu=0.15, dt=0.5, max_iters=500, alpha=np.log(1e6), h=0.5, sigma_max=18.0, sigma_min=4.0, tol=2.0, verbose=True):
    mask = largest_component(np.asarray(mask, bool))
    root = find_root(mask)
    m = medial_function(mask, spacing, mu, dt, max_iters, verbose)
    dist = distance_transform_edt(mask, sampling=spacing)

    travel_time = travel_time_from(mask, root, np.exp(alpha * np.ma.filled(m, 0.0)), spacing)
    reach = travel_time_from(mask, root, dist, spacing)
    paths = trace_paths(mask, travel_time, reach, dist, root, spacing, h, max(np.max(dist), sigma_max), sigma_min)
    return build_tree(paths, tol, spacing)


def largest_component(mask):
    labels, _ = label(mask)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def find_root(mask, depth=10):
    coords = np.argwhere(mask)
    top = coords[:, 2].max() - depth
    root = np.round(coords[coords[:, 2] == top].mean(axis=0)).astype(int)
    if not mask[tuple(root)]:
        root = coords[np.argmin(np.linalg.norm(coords - root, axis=1))]
    return tuple(int(r) for r in root)


def travel_time_from(mask, root, speed, spacing):
    phi = np.full(mask.shape, np.inf)
    phi[root] = 0.0
    phi = np.ma.MaskedArray(phi, mask=~mask)
    return skfmm.travel_time(phi, np.ascontiguousarray(speed), dx=spacing, order=1)


def trace_paths(mask, travel_time, reach, dist, root, spacing, h, sigma_max, sigma_min):
    tracer = Tracer(travel_time, mask, root, spacing, h)
    remaining = np.ma.filled(reach, -1.0)
    remaining[:, :, root[2]:] = -1.0  # only below the root
    paths = []
    while True:
        idx = remaining.argmax()
        if remaining.flat[idx] <= 0:
            break
        path = tracer.trace(np.unravel_index(idx, remaining.shape))
        sigma = np.linspace(sigma_max, sigma_min, len(path))
        vox = np.rint(path).astype(int)
        keep = np.r_[True, np.any(vox[1:] != vox[:-1], axis=1)]  # one sphere per voxel of the path
        radii = dist[tuple(vox[keep].T)] + sigma[keep]
        _cover(remaining, path[keep], radii, spacing)
        paths.append(path)
    return paths


def _cover(remaining, path, radii, spacing):
    sp = np.asarray(spacing, float)
    for p, r in zip(path, radii):
        lo = np.maximum(np.floor(p - r / sp), 0).astype(int)
        hi = np.minimum(np.ceil(p + r / sp) + 1, remaining.shape).astype(int)
        box = tuple(slice(a, b) for a, b in zip(lo, hi))
        d2 = sum(((g - c) * s) ** 2 for g, c, s in zip(np.ogrid[box], p, sp))
        remaining[box][d2 <= r ** 2] = -1.0


def build_tree(paths, tol, spacing):
    tree = BranchTree(spacing)
    _grow(tree, tree.add_branch(None, 0), paths, list(range(len(paths))), 0, tol, np.asarray(spacing, float))
    return tree


def _grow(tree, branch, paths, group, k, tol, sp):
    points = [] if branch.parent is None else [tree.branches[branch.parent].end_point]
    while True:
        group = [i for i in group if k < len(paths[i])]
        if not group:
            break
        clusters = _cluster(paths, group, k, tol, sp)
        if len(clusters) > 1:
            if not points:
                points.append(np.mean([paths[i][k] for i in group], axis=0))
            branch.points = np.array(points)
            for cluster in clusters:
                child = tree.add_branch(branch.id, branch.generation + 1)
                _grow(tree, child, paths, cluster, k, tol, sp)
            return
        points.append(np.mean([paths[i][k] for i in group], axis=0))
        if len(group) == 1:
            points.extend(paths[group[0]][k + 1:])
            break
        k += 1
    branch.points = np.array(points)


def _cluster(paths, group, k, tol, sp):
    pts = {i: paths[i][k] * sp for i in group}
    left = set(group)
    clusters = []
    while left:
        seed = left.pop()
        cluster, stack = [seed], [seed]
        while stack:
            a = stack.pop()
            for b in list(left):
                if np.linalg.norm(pts[a] - pts[b]) < tol:
                    left.discard(b)
                    cluster.append(b)
                    stack.append(b)
        clusters.append(cluster)
    return clusters
