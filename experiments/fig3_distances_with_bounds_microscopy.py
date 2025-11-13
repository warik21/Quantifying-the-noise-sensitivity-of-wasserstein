import matplotlib.pyplot as plt
import numpy as np
import os
from utils.pre_processing import downscale_grayscale_images, calculate_costs, read_dotmark_image
from utils.calculations import run_image_pair_full_experiment

results_dir = os.path.join('results/microscopy_distances_with_bounds')
os.makedirs(results_dir, exist_ok=True)
category = 'MicroscopyImages'
image_numbers = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
resolution_og = 32
resolution = 32
noise_values = np.logspace(start=-7, stop=-1, num=50)
num_exp = 20
n_parallel = 20
images = [read_dotmark_image(category, resolution_og, i) for i in range(1, len(image_numbers) + 1)]
images_curr = downscale_grayscale_images(images, resolution)
cost_matrix = calculate_costs((resolution, resolution), metric='euclidean', cyclic=True)

df = run_image_pair_full_experiment(img1=images_curr[0], img2=images_curr[1],
                                    exp_name='CryoEM_two_images_32_torus_big_range',
                                    cost_matrix=cost_matrix,
                                    noise_std_values=noise_values,
                                    num_exp=num_exp,
                                    n_parallel=n_parallel,
                                    results_dir=results_dir,
                                    force_eval=False)
