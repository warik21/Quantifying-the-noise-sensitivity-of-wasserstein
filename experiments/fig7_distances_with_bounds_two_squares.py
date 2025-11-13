import matplotlib.pyplot as plt
import numpy as np
import os
from utils.pre_processing import downscale_grayscale_images, calculate_costs
from utils.calculations import run_image_pair_full_experiment

results_dir = os.path.join('results/results_two_squares')
generate_sample_images = False
resolution_og = 32
resolution = 32
noise_values = np.logspace(start=-7, stop=-1, num=50)
num_exp = 100
n_parallel = 20

image1 = np.zeros((resolution, resolution))
image2 = np.zeros((resolution, resolution))

image1[8, 8] = 1
image2[24, 24] = 1

images = [image1, image2]

cost_matrix = calculate_costs((resolution, resolution), metric='euclidean', cyclic=True)
df = run_image_pair_full_experiment(img1=images[0], img2=images[1],
                                    exp_name='Point_Squares_two_images_32_torus_big_range',
                                    cost_matrix=cost_matrix,
                                    noise_std_values=noise_values,
                                    num_exp=num_exp,
                                    n_parallel=n_parallel,
                                    results_dir=results_dir,
                                    force_eval=True)
