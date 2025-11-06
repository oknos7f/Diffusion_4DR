# python
import numpy as np
from typing import Optional, Sequence, Tuple

def voxelize(points: np.ndarray,
             voxel_size: Sequence[float] = (1.0, 1.0, 1.0),
             grid_range: Optional[Sequence[float]] = None,
             agg: str = "max") -> np.ndarray:
    """
    Voxelize (N,4) points -> (M,4) where each row is (x_center, y_center, z_center, aggregated_value).
    - points: numpy array shape (N,4): x,y,z,value
    - voxel_size: scalar or (vx,vy,vz)
    - grid_range: optional (xmin,xmax,ymin,ymax,zmin,zmax). If None, derived from points.
    - agg: aggregation method: 'max','min','mean','sum','count','median'
    """
    if not isinstance(points, np.ndarray):
        raise TypeError("points must be numpy.ndarray")
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError("points must have shape (N,4)")
    pts = points.astype(np.float64)
    vals = pts[:, 3].astype(np.float64)
    coords = pts[:, :3]

    # voxel_size to array
    if np.isscalar(voxel_size):
        vs = np.array([voxel_size, voxel_size, voxel_size], dtype=np.float64)
    else:
        vs = np.asarray(voxel_size, dtype=np.float64)
        if vs.shape != (3,):
            raise ValueError("voxel_size must be scalar or sequence of 3 floats")

    # determine grid bounds
    if grid_range is None:
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        # expand slightly to include max points exactly on boundary
        grid_min = mins - 1e-6
        grid_max = maxs + 1e-6
    else:
        gr = np.asarray(grid_range, dtype=np.float64)
        if gr.shape != (6,):
            raise ValueError("grid_range must be (xmin,xmax,ymin,ymax,zmin,zmax)")
        grid_min = np.array([gr[0], gr[2], gr[4]], dtype=np.float64)
        grid_max = np.array([gr[1], gr[3], gr[5]], dtype=np.float64)

    # number of voxels per axis
    n_vox = np.floor((grid_max - grid_min) / vs).astype(int)
    if np.any(n_vox <= 0):
        raise ValueError("Invalid grid range or voxel_size resulting in non-positive voxel counts")

    # voxel indices
    idx = np.floor((coords - grid_min) / vs).astype(int)
    # mask points falling outside (shouldn't if grid_range built from points)
    valid_mask = np.all((idx >= 0) & (idx < n_vox), axis=1)
    if not np.any(valid_mask):
        return np.zeros((0, 4), dtype=np.float64)
    idx = idx[valid_mask]
    vals = vals[valid_mask]

    # linear index for grouping
    nx, ny, nz = n_vox
    lin = (idx[:, 0] * (ny * nz)) + (idx[:, 1] * nz) + idx[:, 2]

    order = np.argsort(lin)
    lin_sorted = lin[order]
    vals_sorted = vals[order]
    idx_sorted = idx[order]

    uniques, start_idx, counts = np.unique(lin_sorted, return_index=True, return_counts=True)

    # aggregation
    if agg == "max":
        agg_vals = np.maximum.reduceat(vals_sorted, start_idx)
    elif agg == "min":
        agg_vals = np.minimum.reduceat(vals_sorted, start_idx)
    elif agg == "sum":
        agg_vals = np.add.reduceat(vals_sorted, start_idx)
    elif agg == "mean":
        sums = np.add.reduceat(vals_sorted, start_idx)
        agg_vals = sums / counts
    elif agg == "count":
        agg_vals = counts.astype(np.float64)
    elif agg == "median":
        # median doesn't have a vectorized reduceat -> compute per group
        agg_vals = np.empty(uniques.shape[0], dtype=np.float64)
        for i, (s, c) in enumerate(zip(start_idx, counts)):
            agg_vals[i] = np.median(vals_sorted[s:s + c])
    else:
        raise ValueError("Unsupported agg method")

    # representative voxel indices (take first index in each group)
    rep_idx = idx_sorted[start_idx]

    # voxel centers
    centers = grid_min + (rep_idx + 0.5) * vs

    result = np.hstack((centers, agg_vals.reshape(-1, 1)))
    return result
