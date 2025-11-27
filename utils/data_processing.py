import numpy as np
from typing import Optional, Sequence
import PIL.Image as Image


def crop_image_half(image: Image.Image, left: bool = True) -> Image.Image:
    """
    PIL Image의 좌/우 절반을 잘라 반환합니다.
    """
    if not hasattr(image, "size"):
        raise TypeError("`image`는 PIL Image 여야 합니다.")
    width, elevation = image.size
    half_width = width // 2  # 2560, 720 -> 1280, 720
    
    if left:
        return image.crop((0, 0, half_width, elevation))
    else:
        return image.crop((half_width, 0, width, elevation))


def polar_to_cartesian(data: np.ndarray,
                       range_bins: int = None,
                       azimuth_bins: int = None,
                       elevation_bins: int = None,
                       num_points: int = 1024,
                       coord_normalize: bool = True) -> np.ndarray:
    """
    polar(거리, 각도, 고도) 3D 그리드 데이터에서 상위 num_points개의 포인트를 추출하여
    Cartesian (N, 4) [x, y, z, power]로 변환.

    threshold 없이 무조건 Power가 높은 순서대로 num_points개를 채웁니다.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("`data`는 numpy.ndarray 여야 합니다.")
    if data.ndim != 4 or data.shape[0] != 2:
        raise ValueError("`data`는 (2, D, T, H) 형태의 4차원 복소수 텐서여야 합니다.")
    
    # 1. Power 데이터 준비
    raw_power = data[0]
    # doppler = data[1] # 현재 미사용
    epsilon = 1e-6
    power = 10 * np.log10(raw_power + epsilon)  # dB 변환
    
    d, t, h = power.shape
    range_val = range_bins or d
    azimuth_val = azimuth_bins or t
    elevation_val = elevation_bins or h
    
    flat_power = power.ravel()
    total_elements = flat_power.size
    
    target_k = min(num_points, total_elements)
    
    if target_k <= 0:
        return np.zeros((0, 4), dtype=np.float32)
    
    top_indices_flat = np.argpartition(flat_power, -target_k)[-target_k:]
    
    sorted_args = np.argsort(flat_power[top_indices_flat])[::-1]
    top_indices_flat = top_indices_flat[sorted_args]
    
    idx_d, idx_t, idx_h = np.unravel_index(top_indices_flat, power.shape)
    valid_power = flat_power[top_indices_flat].astype(np.float32)
    
    p_min = valid_power.min()
    p_max = valid_power.max()
    
    if p_max - p_min == 0:
        valid_power[:] = 0.0
    else:
        valid_power = (valid_power - p_min) / (p_max - p_min)
    
    rho_1d = np.linspace(0.0, range_val - 1.0, range_val)
    r_vec = rho_1d[idx_d]
    
    azimuth_max_deg = (azimuth_val - 1) / 2.0
    azimuth_1d = np.deg2rad(np.linspace(-azimuth_max_deg, azimuth_max_deg, azimuth_val))
    a_vec = azimuth_1d[idx_t]
    
    phi_1d = np.deg2rad(np.linspace(0.0, elevation_val - 1.0, elevation_val))
    p_vec = phi_1d[idx_h]
    
    r_proj = r_vec * np.cos(p_vec)
    x = r_proj * np.cos(a_vec)
    y = r_proj * np.sin(a_vec)
    z = r_vec * np.sin(p_vec)
    
    cartesian_points = np.stack([x, y, z, valid_power], axis=-1)
    
    if coord_normalize:
        x_min, x_max = 0.0, 255.0
        y_min, y_max = -203.7, 203.7
        z_min, z_max = 0.0, 149.9
        
        cartesian_points[:, 0] = (cartesian_points[:, 0] - x_min) / (x_max - x_min) * 2.0 - 1.0
        cartesian_points[:, 1] = (cartesian_points[:, 1] - y_min) / (y_max - y_min) * 2.0 - 1.0
        cartesian_points[:, 2] = (cartesian_points[:, 2] - z_min) / (z_max - z_min) * 2.0 - 1.0
    
    return cartesian_points


def voxelize(points: np.ndarray,
             voxel_size: Sequence[float] = (0.01, 0.01, 0.01),
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