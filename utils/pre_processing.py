import os

import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt
from skimage.transform import resize
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def read_dotmark_image(category: str = 'ClassicImages',
                       resolution: int = 32,
                       image_index: int = 1,
                       base_path: str = Path(os.getenv('DOTMARK_DIR')).resolve()
                       ) -> np.ndarray:
    """
    Reads an image using Pillow and converts it to a NumPy array.

    Parameters:
    - category: str, the category of the image (e.g., 'ClassicImages').
    - resolution: int, the resolution of the image (e.g., 32).
    - image_index: int, the index of the image (1-based).
    - base_path: str, the base path to the DOTmark images directory.

    Returns:
    - image_array: np.ndarray, the grayscale image as a NumPy array.
    """
    filename = f"picture{resolution}_10{image_index:02d}.png"
    full_path = os.path.join(base_path, category, filename)

    with Image.open(full_path) as img:
        return np.array(img.convert('L'))


def noise_image(im, noise_param):
    """
    takes an image and adds zero-mean Gaussian noise with standard deviation noise_param

    Parameters:
    - im: np.ndarray, the input image array.
    - noise_param: float, the standard deviation of the Gaussian noise to be added.

    Returns:
    - noisy_image: np.ndarray, the resulting image after adding noise.
    """
    # Generate a noise array with the same shape as the image
    noise = np.random.normal(0, noise_param, im.shape)
    noise -= np.mean(noise)  # Center the noise around zero

    # Add the noise to the image. This creates a new noisy image array.
    noisy_image = im + noise

    return noisy_image


def noise_and_split_image(image, noise_param) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Adds Gaussian noise to an image and splits it into positive and negative parts.

    Parameters:
    - im: np.ndarray, the input image array.
    - noise_param: float, the standard deviation of the Gaussian noise to be added.

    Returns:
    - noisy_image: np.ndarray, the resulting noisy image.
    - pos_part: np.ndarray, the positive part of the noisy image, \\mu_+ in the papers notation
    - neg_part: np.ndarray, the negative part of the noisy image, \\mu_- in the papers notation
    """
    noisy_image = noise_image(image, noise_param)

    negative_noise = np.zeros(image.shape)
    negative_noise[noisy_image < 0] = -noisy_image[noisy_image < 0]

    noisy_image_pos = noisy_image.copy()
    noisy_image_pos[noisy_image_pos < 0] = 0

    return noisy_image, noisy_image_pos, negative_noise


def calculate_costs(size, metric='euclidean', cyclic=True):
    """
    Calculate the cost matrix for a grid of given size using the specified metric.

    Parameters:
    - size: int or tuple of ints, the dimensions of the grid.
    - metric: str, the distance metric to use ('euclidean' or 'l1').
    - cyclic: bool, whether to treat the grid as cyclic (toroidal).

    Returns:
    - d: np.ndarray, the cost matrix of shape (N, N) where N
    """
    if isinstance(size, int):
        size = (size,)

    ndim = len(size)
    grid = np.indices(size).reshape(ndim, -1).T
    A = grid[:, None, :].astype(float)
    B = grid[None, :, :].astype(float)

    delta = np.abs(A - B)
    if cyclic:
        wrap = np.minimum(delta, np.array(size) - delta)
        diff = wrap / np.array(size)
    else:
        diff = delta / np.array(size)

    if metric == 'euclidean':
        d = np.sqrt((diff ** 2).sum(axis=-1))
    elif metric == 'l1':
        d = diff.sum(axis=-1)
    else:
        d = np.sqrt((diff ** 2).sum(axis=-1))
    return d


def downscale_grayscale_images(images, resolution):
    """
    Downscale a list of grayscale images to a target resolution using proper interpolation.

    Parameters:
    - images: list of np.ndarray, the input grayscale images.
    - resolution: int, the target resolution (assumed square).

    Returns:
    - downscaled_images: list of np.ndarray, the downscaled images.
    """
    downscaled_images = [resize(img, (resolution, resolution),
                                anti_aliasing=True) for img in images]
    normalized_images = [img / np.sum(img) for img in downscaled_images]
    return normalized_images
