import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv, find_dotenv
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from utils.pre_processing import downscale_grayscale_images, read_dotmark_image, \
    noise_image, calculate_costs
from utils.calculations import run_image_pair_full_experiment
load_dotenv(find_dotenv())
# --- Configuration ---
category = 'MicroscopyImages'
results_dir = os.path.join('results/two_image_experiments/CryoEM')
generate_sample_images = False
resolution = 32
image_numbers = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
noise_std_values = np.logspace(start=-4, stop=-2, num=80)  # Standard deviation values
num_exp = 100
n_parallel = 100

# Data Loading and Pre-processing
images = [read_dotmark_image(category, resolution, i) for i in range(1, len(image_numbers) + 1)]
images_curr = downscale_grayscale_images(images, resolution)
cost_matrix_euclidean = calculate_costs((resolution, resolution), metric='euclidean', cyclic=True)
img1 = images_curr[0]
img2 = images_curr[1]

# Run Experiment
experiment_results = run_image_pair_full_experiment(img1=img1, img2=img2,
                                                    exp_name='CryoEM_two_images_32_torus',
                                                    cost_matrix=cost_matrix_euclidean,
                                                    noise_std_values=noise_std_values,
                                                    num_exp=num_exp,
                                                    n_parallel=n_parallel,
                                                    results_dir=results_dir,
                                                    force_eval=False)


all_noise_std_levels = np.logspace(-4, -2, 10)
display_slice = slice(1, -1)
display_x_std = all_noise_std_levels[display_slice]

noisy_images_1_all = [noise_image(img1, nv) for nv in all_noise_std_levels]
noisy_images_2_all = [noise_image(img2, nv) for nv in all_noise_std_levels]
noisy_images_1 = noisy_images_1_all[display_slice]
noisy_images_2 = noisy_images_2_all[display_slice]

fig = plt.figure(figsize=(16, 8))
gs = GridSpec(4, 8, figure=fig, hspace=0.22, wspace=0.05)

ax_strip = fig.add_subplot(gs[0:2, :])
ax_strip.set_xscale('log')
ax_strip.set_xlim(all_noise_std_levels[0], all_noise_std_levels[-1])
ax_strip.set_ylim(-0.05, 1.08)
ax_strip.axis('off')


# Function to add thumbnail images
def add_thumb(ax, img, x, y, zoom=2.6):
    oi = OffsetImage(img, cmap='gray', zoom=zoom)
    ab = AnnotationBbox(
        oi, (x, y),
        xycoords='data',
        frameon=False,
        box_alignment=(0.5, 0.5),
        zorder=3,
        clip_on=False,
        annotation_clip=False
    )
    ax.add_artist(ab)


thumb_zoom = 2.6
row_y_image1 = 0.78
row_y_image2 = 0.22
row_y_titles = 0.99

# Plot the images
for x, im1, im2 in zip(display_x_std, noisy_images_1, noisy_images_2):
    add_thumb(ax_strip, im1, x, row_y_image1, zoom=thumb_zoom)
    add_thumb(ax_strip, im2, x, row_y_image2, zoom=thumb_zoom)

ax_strip.text(-0.05, row_y_image1, 'Image 1', ha='left', va='center', fontsize=12, transform=ax_strip.transAxes)
ax_strip.text(-0.05, row_y_image2, 'Image 2', ha='left', va='center', fontsize=12, transform=ax_strip.transAxes)

ax_ratio = fig.add_subplot(gs[2:, :])
ax_ratio.set_xscale('log')
ax_ratio.set_yscale('log')
ax_ratio.set_xlim(all_noise_std_levels[0], all_noise_std_levels[-1])
ax_ratio.margins(x=0)

# Plot distance series
ax_ratio.plot(experiment_results['noise_std'], experiment_results['W1_ratio'],
              label=r'$W_1 : \frac{W_1(\mu_\sigma, \nu_\sigma)}{W_1(\mu, \nu)}$', color='#b1182b', linewidth=2)
ax_ratio.plot(experiment_results['noise_std'], experiment_results['W2_ratio'],
              label=r'$W_2 : \frac{W_2(\mu_\sigma, \nu_\sigma)}{W_2(\mu, \nu)}$', color='#2065ab', linewidth=2)
ax_ratio.plot(experiment_results['noise_std'], experiment_results['L2_ratio'],
              label=r'$L_2 : \frac{\|\mu_\sigma - \nu_\sigma\|_2}{\|\mu - \nu\|_2}$', color='sandybrown', linewidth=2)
ax_ratio.set_xticks(display_x_std)
ax_ratio.set_xticklabels([f'{nv:.0e}' for nv in display_x_std], rotation=45, ha='right', fontsize=12)

# Interpolated scatters at those x’s
scatter_y_w1 = np.interp(display_x_std, experiment_results['noise_std'], experiment_results['W1_ratio'])
scatter_y_w2 = np.interp(display_x_std, experiment_results['noise_std'], experiment_results['W2_ratio'])
scatter_y_l2 = np.interp(display_x_std, experiment_results['noise_std'], experiment_results['L2_ratio'])
ax_ratio.scatter(display_x_std, scatter_y_w1, s=55, zorder=5, edgecolor='black', c='#b1182b', marker='o')
ax_ratio.scatter(display_x_std, scatter_y_w2, s=55, zorder=5, edgecolor='black', c='#2065ab', marker='s')
ax_ratio.scatter(display_x_std, scatter_y_l2, s=55, zorder=5, edgecolor='black', c='sandybrown', marker='^')

ax_ratio.set_xlabel(r'$(\sigma)$', fontsize=12)
ax_ratio.set_ylabel(r'Distance Ratio', fontsize=12)
ax_ratio.set_title(r'Distance Ratios vs. Noise Standard Deviation', fontsize=14)
ax_ratio.legend(fontsize=16, loc='best')
ax_ratio.grid(True, axis='y', linestyle='-', alpha=0.5)

# Vertical guides at each thumbnail center
for x in display_x_std:
    ax_ratio.axvline(x=x, linestyle='--', linewidth=0.6, alpha=0.7)

plt.tight_layout(rect=[0.06, 0, 1, 1])
plt.show()
fig.savefig(os.path.join('results/noisy_images_and_ratios.pdf'), bbox_inches='tight')
