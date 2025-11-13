import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import gridspec

from utils.pre_processing import calculate_costs, downscale_grayscale_images, noise_image
from utils.calculations import measure_noise_effects_on_image_pair, run_image_pair_experiment

images_path = "results/rotation experiment/extracted_images_normalized"
results_dir = 'results_gradual_change_images'
resolution = 32
num_exp = 50
n_parallel = 20
force_eval = True
full_path = os.path.join(os.getcwd(), images_path)

images = os.listdir(full_path)
images = [img for img in images if img.endswith('.png')]
images.sort()
images = [np.array(Image.open(os.path.join(full_path, img)).convert('L')) for img in images]
images_curr = downscale_grayscale_images(images, resolution)
images_curr = images_curr[::2]

for i, image in enumerate(images_curr):
    image = np.pad(image, ((16, 16), (16, 16)), mode='constant', constant_values=0)
    image_cropped = image[i + 6:i + 38, i + 6:i + 38]
    images_curr[i] = image_cropped / image_cropped.sum()

# Show all the images in a 4x5 grid
fig, axs = plt.subplots(4, 5, figsize=(10, 8))
for i, ax in enumerate(axs.flat):
    if i < len(images_curr):
        ax.imshow(images_curr[i], cmap='gray')
        ax.set_title(f'Image {i}')
    ax.axis('off')
plt.tight_layout()

metrics = ['L2', 'W1', 'W2']
cost_matrix = calculate_costs((resolution, resolution), metric='euclidean', cyclic=True)

noisy_images = []
noise_param = 0.01

if force_eval:
    results = np.zeros((len(images_curr), len(images_curr), len(metrics)))
    results_noisy = np.zeros((len(images_curr), len(images_curr), len(metrics)))

    for i, image in enumerate(images_curr):
        noised_image = noise_image(image, noise_param=noise_param)
        noisy_images.append(noised_image)
        for j, image2 in enumerate(images_curr):
            if i < j:
                continue
            results_df = run_image_pair_experiment(image, image2,
                                                   cost_matrix=cost_matrix,
                                                   noise_std_values=[noise_param],
                                                   num_exp=num_exp,
                                                   n_parallel=n_parallel,
                                                   force_eval=True)

            if np.array_equal(image, image2) and i != j:
                print("Same image, though different instances:", i, j)
            results_noisy[i, j, 0] = results_df['noisy_vs_noisy_L2'].values[0]
            results_noisy[j, i, 0] = results_noisy[i, j, 0]
            results_noisy[i, j, 1] = results_df['noisy_vs_noisy_W1'].values[0]
            results_noisy[j, i, 1] = results_noisy[i, j, 1]  # Symmetric
            results_noisy[i, j, 2] = results_df['noisy_vs_noisy_W2'].values[0]
            results_noisy[j, i, 2] = results_noisy[i, j, 2]

            results[i, j, 0] = results_df['original_L2'].values[0]
            results[j, i, 0] = results[i, j, 0]
            results[i, j, 1] = results_df['original_W1'].values[0]
            results[j, i, 1] = results[i, j, 1]  # Symmetric
            results[i, j, 2] = results_df['original_W2'].values[0]
            results[j, i, 2] = results[i, j, 2]

    # Save the results
    np.save(os.path.join('results/rotation experiment', 'distances_between_images_turning_and_moving_noise_01.npy'),
            results)
    np.save(
        os.path.join('results/rotation experiment', 'distances_between_noisy_images_turning_and_moving_noise_01.npy'),
        results_noisy)

else:
    results = np.load(os.path.join('results/rotation experiment temp', 'distances_between_images_turning_and_moving_noise_02.npy'))
    results_noisy = np.load(
        os.path.join('results/rotation experiment temp', 'distances_between_noisy_images_turning_and_moving_noise_02.npy'))


pair_wspaces = [0.1, 0.1, 0.1]

fig = plt.figure(figsize=(15, 8))
outer = gridspec.GridSpec(1, 3, figure=fig, wspace=0.25)

axs = []
titles = ['L2', 'W1', 'W2']
for i, (title, w) in enumerate(zip(titles, pair_wspaces)):
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[i], wspace=w
    )
    ax_noiseless = fig.add_subplot(inner[0])
    ax_noisy = fig.add_subplot(inner[1])
    axs.extend([ax_noiseless, ax_noisy])

for k, t in enumerate(titles):
    axs[2*k].imshow(results[:, :, k], cmap='RdBu_r', interpolation='nearest')
    axs[2*k+1].imshow(results_noisy[:, :, k], cmap='RdBu_r', interpolation='nearest')
    axs[2*k].set_title(f'{t} noiseless', fontsize=12, pad=15)
    axs[2*k+1].set_title(f'{t} noisy', fontsize=12, pad=15)

for ax in axs:
    ax.axis('off')
plt.tight_layout()

if not os.path.exists('results/rotation experiment'):
    os.makedirs('results/rotation experiment')
file_name = os.path.join('results/rotation experiment',
                         'original_distances_vs_noisy_distances_between_images_turning_and_moving_noise_02_PuOr_r_5exp.pdf')
plt.savefig(file_name, format='pdf', dpi=1200, bbox_inches='tight')
plt.show()


noisy_images = []
for i, image in enumerate(images_curr):
    noised_image = noise_image(image, noise_param=noise_param)
    noisy_images.append(noised_image)
fig, axs = plt.subplots(4, 10, figsize=(20, 8))
for i in range(4):
    for j in range(10):
        if j < 5:
            axs[i, j].imshow(images_curr[i * 5 + j], cmap='gray')
        else:
            axs[i, j].imshow(noisy_images[i * 5 + (j - 5)], cmap='gray')
        axs[i, j].axis('off')
axs[0, 2].set_title('Original Images', fontsize=20)
axs[0, 7].set_title('Noisy Images', fontsize=20)
plt.tight_layout()
file_name = os.path.join('results/rotation experiment', 'images_object_side_by_side_turning_and_moving_noise_01.pdf')
plt.savefig(file_name, format='pdf', dpi=1200, bbox_inches='tight')
plt.show()