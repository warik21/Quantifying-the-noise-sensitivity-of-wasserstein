import numpy as np
import matplotlib.pyplot as plt
import os
from joblib import Parallel, delayed
import ot

from utils.pre_processing import calculate_costs, noise_and_split_image


results_dir = os.path.join('results/results_two_squares')
os.makedirs(results_dir, exist_ok=True)

resolution_og = 32
resolution = 32
num_exp = 20
n_parallel = 20
noise_values = [0, 0.0013, 0.002, 0.004, 0.008, 0.016]
image1 = np.zeros((resolution, resolution))
image2 = np.zeros((resolution, resolution))
src_r, src_c = 8, 8
tgt_r, tgt_c = 24, 24
image1[src_r, src_c] = 1.0
image2[tgt_r, tgt_c] = 1.0
images = [image1, image2]
cost_matrix = calculate_costs((resolution, resolution), metric='euclidean', cyclic=True)


def _one_experiment(noise):
    noisy_original1, pos1, neg1 = noise_and_split_image(image1, noise)
    noisy_original2, pos2, neg2 = noise_and_split_image(image2, noise)

    a = (pos1 + neg2).flatten()
    b = (pos2 + neg1).flatten()
    P = (ot.emd(a, b, cost_matrix ** 2)) ** 0.5

    moved = P[src_r * resolution + src_c, :].reshape((resolution, resolution))
    received = P[:, tgt_r * resolution + tgt_c].reshape((resolution, resolution))
    return moved, received


avg_moved_list = []
avg_received_list = []

for noise in noise_values:
    results = Parallel(n_jobs=n_parallel, verbose=0)(
        delayed(_one_experiment)(noise) for _ in range(num_exp)
    )
    moved_stack = np.stack([r[0] for r in results], axis=0)
    recv_stack = np.stack([r[1] for r in results], axis=0)

    avg_moved = moved_stack.mean(axis=0)
    avg_recv = recv_stack.mean(axis=0)

    avg_moved_list.append(avg_moved)
    avg_received_list.append(avg_recv)

avg_moved_arr = np.stack(avg_moved_list, axis=0)
avg_received_arr = np.stack(avg_received_list, axis=0)

np.save(os.path.join(results_dir, 'avg_mass_moved_from_8_8.npy'), avg_moved_arr)
np.save(os.path.join(results_dir, 'avg_mass_received_at_24_24.npy'), avg_received_arr)
np.save(os.path.join(results_dir, 'noise_values.npy'), noise_values)

num_cols = len(noise_values)
fig, axes = plt.subplots(2, num_cols, figsize=(2.8 * num_cols, 6), constrained_layout=True)
axes[0, 0].text(-0.1, 0.5, "from source", rotation=90,
                va='center', ha='center', transform=axes[0, 0].transAxes, fontsize=20)
axes[1, 0].text(-0.1, 0.5, "to target", rotation=90,
                va='center', ha='center', transform=axes[1, 0].transAxes, fontsize=20)
for j, noise in enumerate(noise_values):
    ax_top = axes[0, j]
    im_top = ax_top.imshow(avg_moved_arr[j], cmap='gray')
    ax_top.axis('off')
    ax_top.set_title(f'$\\sigma$={noise:.2e}', fontsize=20)

    ax_bot = axes[1, j]
    im_bot = ax_bot.imshow(avg_received_arr[j], cmap='gray')
    ax_bot.axis('off')

plt.savefig(os.path.join(results_dir, 'avg_mass_flow_noise_sweep.pdf'), dpi=1200, bbox_inches='tight')
plt.show()
