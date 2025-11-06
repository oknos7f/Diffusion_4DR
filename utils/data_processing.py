import numpy as np
import PIL.Image as Image


def crop_image_half(image: Image, left=False) -> Image:
    width, height = img.size
    half_width = width // 2
    
    if left:
        return image.crop((0, 0, half_width, height))
    else:
        return image.crop((half_width, 0, width, height))
    

def normalization(data: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    mask = data[:, 3] >= threshold
    arr = data[mask]
    
    if arr.shape[0] == 0:
        raise Exception("wrong threshold: " + str(threshold))
    
    power = arr[:, 3]
    p_min = np.min(power)
    p_max = np.max(power)
    
    print(f"p_min: {p_min}, p_max: {p_max}")
    
    if p_max - p_min == 0:
        arr[:, 3] = 0.0
    else:
        arr[:, 3] = (power - p_min) / (p_max - p_min)
    
    return arr


def polar_to_cartesian(data: np.ndarray,
                       distance=256, theta=107, height=37, threshold=0.0) -> np.ndarray:
    theta_max_deg = theta // 2
    rho_1d = np.linspace(0.0, distance - 1, distance)
    theta_1d = np.deg2rad(np.linspace(theta_max_deg, -theta_max_deg + 1, theta))
    phi_1d = np.deg2rad(np.linspace(0, height - 1, height))
    
    rho_grid, theta_grid, phi_grid = np.meshgrid(rho_1d, theta_1d, phi_1d, indexing='ij')
    
    r_proj = rho_grid * np.cos(phi_grid)
    x = r_proj * np.cos(theta_grid)
    y = r_proj * np.sin(theta_grid)
    z = rho_grid * np.sin(phi_grid)
    
    grid_shape = x.shape
    num_points = np.prod(grid_shape)
    
    cartesian_points = np.stack([x.flatten(),
                                 y.flatten(),
                                 z.flatten(),
                                 data.flatten()[:num_points]], axis=-1)
    
    if threshold:
        return normalization(cartesian_points, threshold=threshold)
    else:
        return cartesian_points
    

if __name__ == "__main__":
    image_path = "../dataset/data/images/0100033.png"
    img = Image.open(image_path)
    print(img.size) # (2560, 720)
    result = crop_image_half(img)
    print(result.size) # (1280, 720)
    result.show()
