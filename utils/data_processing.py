import numpy as np
from typing import Optional, Sequence, Tuple
import PIL.Image as Image


# example usage:
# data = np.load(data_path, mmap_mode='r')
# data = ct.polar_to_cartesian(data, threshold=99)
# data = ct.voxelize(data, agg='max')


def crop_image_half(image: Image.Image, left: bool = True) -> Image.Image:
    """
    PIL Image의 좌/우 절반을 잘라 반환합니다.
    """
    if not hasattr(image, "size"):
        raise TypeError("`image`는 PIL Image 여야 합니다.")
    width, height = image.size
    half_width = width // 2  # 2560, 720 -> 1280, 720
    
    if left:
        return image.crop((0, 0, half_width, height))
    else:
        return image.crop((half_width, 0, width, height))


def normalization_value(points: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError("입력은 (N, >=4) 형태의 numpy 배열이어야 합니다.")
    
    # print(threshold)
    
    # 1. Threshold 적용 (threshold 이상인 포인트만 남김)
    mask_threshold = points[:, 3] >= threshold
    arr = points[mask_threshold].copy()
    
    original_count = arr.shape[0]
    
    if original_count == 0:
        raise ValueError(f"threshold 값이 너무 높습니다. (남은 포인트 0개): {threshold}")
    
    power = arr[:, 3].astype(np.float32)
    
    # 정규화 (Min-Max Scaling)
    p_min = power.min()
    p_max = power.max()
    
    if p_max - p_min == 0:
        arr[:, 3] = 0.0
    else:
        # Min-Max 정규화: (power - min) / (max - min)
        arr[:, 3] = (power - p_min) / (p_max - p_min)
    
    return arr


def polar_to_cartesian(data: np.ndarray,
                       distance: int = None,
                       theta: int = None,
                       height: int = None,
                       threshold: float = 99,
                       coord_normalize: bool = False) -> np.ndarray:
    """
    polar(거리, 각도, 고도) 형태의 3D 그리드 데이터를 Cartesian 포인트 클라우드 (N,4)로 변환.
    - data: numpy ndarray, shape (D, T, H) 또는 (distance, theta, height)
    
    **NOTE on `coord_normalize` is `False`:**
    The output matrix from `dpolar_to_cartesian` has the following theoretical value ranges:
    
    * **N (Integer):** [0, 10136]  # Number of Points
    * **x (float32):** [0.0, 255.0]  # Forward coordinate
    * **y (float32):** [-203.7, 203.7] # Horizontal coordinate
    * **z (float32):** [0.0, 149.9]  # Vertical coordinate
    * **value (float32):** [0.0, 1.0] # Value Normalized
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("`data`는 numpy.ndarray 여야 합니다.")
    if data.ndim != 4 or data.shape[0] != 2:
        raise ValueError("`data`는 (2, D, T, H) 형태의 4차원 복소수 텐서여야 합니다.")
    
    data_mag = np.sqrt(data[0] ** 2 + data[1] ** 2)
    epsilon = 1e-6
    data_mag = 10 * np.log10(data_mag + epsilon)
    
    threshold_dB = np.percentile(data_mag.flatten(), threshold)
    
    d, t, h = data_mag.shape
    distance = distance or d
    theta = theta or t
    height = height or h

    # 1D 좌표 생성 (인덱스를 각도로 또는 거리 단위로 해석)
    rho_1d = np.linspace(0.0, distance - 1.0, distance)
    # theta를 양/음 대칭으로 배치 (degrees)
    theta_max_deg = (theta - 1) / 2.0
    theta_1d = np.deg2rad(np.linspace(-theta_max_deg, theta_max_deg, theta))
    # phi는 0..height-1을 degree로 간주 (원래 코드 유지)
    phi_1d = np.deg2rad(np.linspace(0.0, height - 1.0, height))

    rho_grid, theta_grid, phi_grid = np.meshgrid(rho_1d, theta_1d, phi_1d, indexing='ij')

    # 구면 -> 직교 투영
    r_proj = rho_grid * np.cos(phi_grid)
    x = r_proj * np.cos(theta_grid)
    y = r_proj * np.sin(theta_grid)
    z = rho_grid * np.sin(phi_grid)

    grid_shape = x.shape
    num_points = int(np.prod(grid_shape))

    data_flat = data_mag.flatten()
    if data_flat.shape[0] < num_points:
        raise ValueError("입력 데이터의 크기가 좌표 그리드 크기보다 작습니다.")

    values = data_mag.flatten()

    cartesian_points = np.stack([
        x.flatten(),
        y.flatten(),
        z.flatten(),
        values
    ], axis=-1)
    
    if coord_normalize:
        # 좌표 정규화 (-1 ~ 1)
        x_min, x_max = 0.0, 255.0
        y_min, y_max = -203.7, 203.7
        z_min, z_max = 0.0, 149.9
        
        cartesian_points[:, 0] = ((cartesian_points[:, 0] - x_min) / (x_max - x_min)) * 2.0 - 1.0
        cartesian_points[:, 1] = ((cartesian_points[:, 1] - y_min) / (y_max - y_min)) * 2.0 - 1.0
        cartesian_points[:, 2] = ((cartesian_points[:, 2] - z_min) / (z_max - z_min)) * 2.0 - 1.0
        
    if threshold and threshold > 0.0:
        return normalization_value(cartesian_points, threshold=threshold_dB)
    else:
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
