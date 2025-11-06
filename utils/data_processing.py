import numpy as np


def normalization(points: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError("입력은 (N, >=4) 형태의 numpy 배열이어야 합니다.")
    
    print(threshold)
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
    print(f"최종 포인트: {arr.shape[0]}개")
    
    return arr


def polar_to_cartesian(data: np.ndarray,
                       distance: int = None,
                       theta: int = None,
                       height: int = None,
                       threshold: float = 0.3) -> np.ndarray:
    """
    polar(거리, 각도, 고도) 형태의 3D 그리드 데이터를 Cartesian 포인트 클라우드 (N,4)로 변환.
    - data: numpy ndarray, shape (D, T, H) 또는 (distance, theta, height)
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

    if threshold and threshold > 0.0:
        return normalization(cartesian_points, threshold=threshold_dB)
    else:
        return cartesian_points
