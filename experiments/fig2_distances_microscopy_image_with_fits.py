import numpy as np
import os
from dotenv import load_dotenv, find_dotenv
from utils.calculations import run_single_image_full_experiment
from utils.pre_processing import downscale_grayscale_images, read_dotmark_image, \
    calculate_costs

load_dotenv(find_dotenv())

category = 'MicroscopyImages'
results_dir = os.path.join('results/two_image_experiments/CryoEM')
generate_sample_images = False
resolution_og = 32
resolution = 32
image_numbers = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
noise_values = np.logspace(start=-7, stop=0, num=40)
num_exp = 100
n_parallel = 20
images = [read_dotmark_image(category, resolution_og, i) for i in range(1, len(image_numbers) + 1)]
images_curr = downscale_grayscale_images(images, resolution)
cost_matrix = calculate_costs((resolution, resolution), metric='euclidean', cyclic=True)

run_single_image_full_experiment(
    images_list=images_curr,
    exp_name=f"CryoEM_torus_lean_single_image_{resolution}_torus_lean",
    cost_matrix=cost_matrix,
    noise_std_values=noise_values,
    num_exp=num_exp,
    n_parallel=n_parallel,
    results_dir=results_dir,
    force_eval=True
)
