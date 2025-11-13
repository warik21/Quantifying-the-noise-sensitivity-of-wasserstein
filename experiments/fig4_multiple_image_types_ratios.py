import numpy as np
import os

from utils.plotting import plot_multiple_distance_ratios
from utils.pre_processing import downscale_grayscale_images, read_dotmark_image, calculate_costs
from utils.calculations import run_image_pair_full_experiment


dotmark_categories = [
    'WhiteNoise',
    'MicroscopyImages',
    'ClassicImages'
]

categories = [
    'WhiteNoise',
    'MicroscopyImages',
    'ClassicImages',
    'TwoSquares'
]

resolution_og = 32
resolution = 32
image_numbers = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
noise_values = np.logspace(start=-7, stop=-1, num=15)
num_exp = 100
n_parallel = 20
cost_matrix = calculate_costs((resolution, resolution), metric='euclidean', cyclic=True)
df_dict = {}

for category in categories:
    results_dir = os.path.join(f'results/two_image_experiments/{category}')
    full_path = os.path.join(os.getcwd(), results_dir)
    if category in dotmark_categories:
        images = [read_dotmark_image(category, resolution_og, i) for i in range(1, len(image_numbers) + 1)]
        images_curr = downscale_grayscale_images(images, resolution)
    else:
        image1 = np.zeros((resolution, resolution))
        image2 = np.zeros((resolution, resolution))

        image1[8, 8] = 1
        image2[24, 24] = 1

        images_curr = [image1, image2]

    df_dict[f'{category}'] = run_image_pair_full_experiment(img1=images_curr[0], img2=images_curr[1],
                                                            exp_name=f'{category}_two_images_32_torus_exp',
                                                            cost_matrix=cost_matrix,
                                                            noise_std_values=noise_values,
                                                            num_exp=num_exp,
                                                            n_parallel=n_parallel,
                                                            results_dir=results_dir + 'all_distances',
                                                            force_eval=False)

results_dfs = [
    df_dict['WhiteNoise'],
    df_dict['ClassicImages'],
    df_dict['MicroscopyImages'],
    df_dict['TwoSquares']
]

# 2. Define titles for each subplot
titles = [
    'White Noise Images',
    'Classic Images',
    'Cryo-EM Images',
    'Two Squares Images'
]

# 3. Call the plotting function
plot_multiple_distance_ratios(
    results_dfs=results_dfs,
    subplot_titles=titles,
    file_prefix='results/',
    save_fig=True
)
