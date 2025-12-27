import random

import numpy as np
import pandas as pd
import math
import os
import ot

from typing import Dict, Any

from joblib import Parallel, delayed

from utils.plotting import plot_distance_ratios, plot_distance_differences, visualize_single_image_results

from utils.pre_processing import noise_and_split_image


def run_image_pair_full_experiment(
        img1: np.ndarray,
        img2: np.ndarray,
        exp_name: str,
        cost_matrix: np.ndarray,
        noise_std_values: np.ndarray,
        num_exp: int = 10,
        n_parallel: int = 5,
        results_dir: str = 'results',
        force_eval: bool = False,
) -> pd.DataFrame:
    """
    Runs a full experiment comparing two images under various noise conditions.

    Args:
        img1 (np.ndarray): First image to compare.
        img2 (np.ndarray): Second image to compare.
        exp_name (str): Name of the experiment.
        cost_matrix (np.ndarray): Cost matrix to use for distance calculations.
        noise_std_values (np.ndarray): Array of noise standard deviations (σ) to test.
        num_exp (int): Number of experiments to run for each noise value.
        n_parallel (int): Number of parallel jobs to run.
        results_dir (str): Directory to save results.
        force_eval (bool): If True, forces re-evaluation even if results exist.
    """
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{exp_name}_results.csv")
    file_prefix = f"{results_dir}/{exp_name}_image_pair_torus_cost_matrix"

    # Run the experiment
    if force_eval:
        print(f"Running {exp_name} with torus cost matrix...")
        results = Parallel(n_jobs=n_parallel)(
            delayed(measure_noise_effects_on_image_pair)(img1, img2, noise_std, cost_matrix, num_exp)
            for noise_std in noise_std_values
        )
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_path, index=False)
    else:
        print(f"Loading existing results from {results_path}")
        results_df = pd.read_csv(results_path)

    # Visualize results
    plot_distance_ratios(df=results_df, file_prefix=file_prefix)

    plot_distance_differences(df=results_df, file_prefix=file_prefix)

    return results_df


def measure_noise_effects_on_image_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    noise_std: float,
    cost_matrix: np.ndarray,
    num_exp: int = 10
) -> Dict[str, float | int]:
    """
    Measures the effect of noise on the Wasserstein and L2 distances between two images.

    Args:
        img1: First image
        img2: Second image
        noise_std: Standard deviation of the noise (σ)
        cost_matrix: Cost matrix to use for distance calculations
        num_exp: Number of noisy experiments to average over

    Returns:
        Dictionary of distances and ratios
    """
    base_distances: Dict[str, float] = calculate_wasserstein_l2_distances(img1, img2, cost_matrix)

    # Run noisy experiments
    noisy_distances = [compare_noisy_image_pairs(img1, img2, noise_std, cost_matrix)
                       for _ in range(num_exp)]

    return {
        'noise_std': noise_std,
        'original_W1': base_distances['w1_distance'],
        'original_W2': base_distances['w2_distance'],
        'original_W3': base_distances['w3_distance'],
        'original_L2': base_distances['l2_distance'],
        'noisy_vs_noisy_W1': np.mean([d['w1_distance'] for d in noisy_distances]),
        'noisy_vs_noisy_W2': np.mean([d['w2_distance'] for d in noisy_distances]),
        'noisy_vs_noisy_W3': np.mean([d['w3_distance'] for d in noisy_distances]),
        'noisy_vs_noisy_L2': np.mean([d['l2_distance'] for d in noisy_distances]),
        'W1_ratio': (np.mean([d['w1_distance'] for d in noisy_distances])) / base_distances['w1_distance'],
        'W2_ratio': (np.mean([d['w2_distance'] for d in noisy_distances])) / base_distances['w2_distance'],
        'W3_ratio': (np.mean([d['w3_distance'] for d in noisy_distances])) / base_distances['w3_distance'],
        'L2_ratio': (np.mean([d['l2_distance'] for d in noisy_distances])) / base_distances['l2_distance'],
        'resolution': img1.shape[0]
    }


def calculate_wasserstein_l2_distances(image1: np.ndarray,
                                       image2: np.ndarray,
                                       cost_matrix: np.ndarray) -> Dict[str, float]:
    """
    Computes Wasserstein distances (W1, W2) and L2 distance between two images.

    Args:
        image1: First image array
        image2: Second image array
        cost_matrix: Cost matrix

    Returns:
        Dictionary with Wp distances and L2 distance
    """
    flat1 = image1.flatten()
    flat2 = image2.flatten()
    cost_base = cost_matrix
    # Calculate L2 distance once to avoid re-computation
    l2_dist = np.linalg.norm(flat1 - flat2)

    return {
        "w1_distance": calculate_w_p_distances(flat1, flat2, 1, cost_base),
        "w2_distance": calculate_w_p_distances(flat1, flat2, 2, cost_base),
        "w3_distance": calculate_w_p_distances(flat1, flat2, 3, cost_base),
        "l2_distance": l2_dist,
    }


def calculate_w_p_distances(flat1: np.ndarray, flat2: np.ndarray, p: int = 1, cost_base=None
                            ) -> Dict[str, float]:
    """
    Computes Wp distance between two images.
    Args:
        flat1: First image array
        flat2: Second image array
        p: Order of the Wasserstein distance
        cost_base: Base cost matrix
    Returns:
        Dictionary with Wp distance
    """
    cost_p = ot.emd2(flat1, flat2, cost_base ** p)
    return cost_p ** (1 / p)


def compare_noisy_image_pairs(img1: np.ndarray, img2: np.ndarray, noise_std: float,
                              cost_matrix: np.ndarray) -> dict:
    """
    Processes and compares two images under noise conditions.

    Args:
        img1: First image
        img2: Second image
        noise_std: Standard deviation of the noise (σ)
        cost_matrix: Cost matrix to use for distance calculations

    Returns:
        Dictionary of distances between original, noisy, and cross-comparisons.
    """
    # Generate noisy versions
    noisy_original1, pos1, neg1 = noise_and_split_image(img1, noise_std)
    noisy_original2, pos2, neg2 = noise_and_split_image(img2, noise_std)

    # The W^\pm distance between the two noisy images
    noisy_distances = calculate_wasserstein_l2_distances(pos1 + neg2, pos2 + neg1, cost_matrix)

    return noisy_distances


def run_single_image_full_experiment(images_list, exp_name, cost_matrix, noise_std_values,
                                     num_exp, n_parallel, results_dir, force_eval=False):
    """
    Run a complete experiment for a single image with both normal and torus cost matrices.

    Args:
        images_list: List of images to use in the experiment.
        exp_name: Name for the experiment (used in filenames).
        cost_matrix: Tuple containing (L1_cost_matrix, L2_cost_matrix).
        noise_std_values: Array of noise standard deviation values (σ) to test.
        num_exp: Number of experiments per noise level.
        n_parallel: Number of parallel jobs.
        results_dir: Directory to save results.
        force_eval: Decides whether to force evaluation or not.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Construct file prefix and output path
    file_suffix = f"single_image_torus_cost_matrix"
    file_prefix = f"{results_dir}/{exp_name}_{file_suffix}"
    output_path = f"{file_prefix}.csv"

    # Run the experiment
    if force_eval:
        print(f"Running {exp_name} with torus cost matrix...")
        results = Parallel(n_jobs=n_parallel)(
            delayed(measure_noise_effect_on_image)(images_list, noise_std, cost_matrix, num_exp)
            for noise_std in noise_std_values
        )
        results_df = pd.DataFrame(results)
        # Save results df to a csv file
        results_df.to_csv(output_path, index=False)
    else:
        print("Loading existing results")
        results_df = pd.read_csv(output_path)

    # Visualize results
    print(f"Visualizing results for {exp_name} with torus cost matrix...")
    # Convert last std value to variance for threshold
    visualize_single_image_results(results_df, file_prefix)

    return results_df


def measure_noise_effect_on_image(images_list: list[np.ndarray],
                                  noise_std: float,
                                  cost_matrix: np.ndarray,
                                  num_exp: int = 10,
                                  output_dir: str = "noisy_images_sampled") -> dict:
    """
    Compares an image with its noisy version using W1, W2, and L2 metrics.

    Args:
        images_list: List of images to sample from
        noise_std: Standard deviation of the noise (σ)
        cost_matrix: Precomputed cost matrix
        num_exp: Number of experiments to run
        output_dir: Directory to save noisy images

    Returns:
        Dictionary containing noise standard deviation (σ) and averaged distances
    """
    # Run experiments
    os.makedirs(output_dir, exist_ok=True)

    # For metrics: run multiple trials
    noisy_distances = [
        compute_noisy_image_distances(images_list, noise_std, cost_matrix)
        for _ in range(num_exp)
    ]

    return {
        'noise_std': noise_std,
        'W1_distance_avg': np.mean([d['w1_distance'] for d in noisy_distances]),
        'W2_distance_avg': np.mean([d['w2_distance'] for d in noisy_distances]),
        'L2_distance_avg': np.mean([d['l2_distance'] for d in noisy_distances]),
        'resolution': images_list[0].shape[0]
    }


def compute_noisy_image_distances(images_array: list[np.ndarray],
                                  noise_std: float,
                                  cost_matrix: np.ndarray) -> dict:
    """
    Computes distances between an image and its noisy version.

    Args:
        images_array: List of images to sample from
        noise_std: Standard deviation of the noise (σ)
        cost_matrix: Precomputed cost matrix

    Returns:
        Dictionary containing W1, W2 and L2 distances
    """
    image = random.sample(images_array, 1)[0]
    noisy_image, pos, neg = noise_and_split_image(image, noise_std)
    distances = calculate_wasserstein_l2_distances(image+neg, pos, cost_matrix)
    return distances


def run_image_pair_experiment(
        img1: np.ndarray,
        img2: np.ndarray,
        noise_std_values: np.ndarray,
        cost_matrix: np.ndarray,
        num_exp: int = 10,
        n_parallel: int = 5,
) -> pd.DataFrame:
    """
    Runs an experiment comparing two images under various noise conditions.

    Args:
        img1 (np.ndarray): First image to compare.
        img2 (np.ndarray): Second image to compare.
        noise_std_values (np.ndarray): Array of noise standard deviations (σ) to test.
        cost_matrix (np.ndarray): Cost matrix to compare.
        num_exp (int): Number of experiments to run for each noise value.
        n_parallel (int): Number of parallel jobs to run.

    Returns:
        pd.DataFrame: DataFrame containing the results with noise variance (σ²) values.
    """
    results = Parallel(n_jobs=n_parallel)(
        delayed(measure_noise_effects_on_image_pair)(img1, img2, noise_std, cost_matrix, num_exp)
        for noise_std in noise_std_values
    )

    df_results = pd.DataFrame(results)

    return df_results
