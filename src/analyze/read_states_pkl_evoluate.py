import os
import pickle
import pickletools
import numpy as np
import json 
import matplotlib.pyplot as plt
from itertools import chain
from src.data_loader import load_sdf_from_csv, vel_read, load_sim_sdf
from src.a_star.Astar_ani import astar_anisotropic, compute_v, contour_2D

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

file = 'grid_search/8/result_evaluation_retina__traj.pkl'
file_bis = 'grid_search/9/result_evaluation_retina__traj.pkl'
with open(file, "rb") as f:
    successful_trajectories_1 = pickle.load(f)
with open(file_bis, "rb") as f:
    successful_trajectories_bis = pickle.load(f)

successful_trajectories = successful_trajectories_1 + successful_trajectories_bis
print(f"Number of successful trajectories: {len(successful_trajectories)}")
print(f"Points per trajectory: {len(successful_trajectories[0])}")

ratio = 5
sdf_func, velocity_retina, x_phys, y_phys, physical_width, physical_height, scale = load_sim_sdf(ratio)
X, Y = np.meshgrid(x_phys, y_phys)
obstacle_contour = contour_2D(sdf_func, X, Y, scale)

fig1, ax1 = plt.subplots(figsize=(10, 10))

ax1.scatter(obstacle_contour[:, 0], obstacle_contour[:, 1], 
           color="#2C2C2C", s=0.4, alpha=0.9, rasterized=True)

sample_trajectories = successful_trajectories
colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(sample_trajectories)))

for i, trajectory in enumerate(sample_trajectories):
    traj = np.array(trajectory)
    ax1.plot(traj[:, 0], traj[:, 1], alpha=0.4, color=colors[i], 
            linewidth=0.8, rasterized=True)
sample_trajectories = successful_trajectories[::5]

for i, trajectory in enumerate(sample_trajectories):
    traj = np.array(trajectory)
    ax1.plot(traj[0, 0], traj[0, 1], marker='o', color="#81E7B4", 
            markersize=6, alpha=0.8, markeredgecolor='white', markeredgewidth=0.8)
    ax1.plot(traj[-1, 0], traj[-1, 1], marker='o', color="#2940E9", 
            markersize=5)

ax1.set_aspect('equal')
ax1.set_xlim(x_phys.min(), x_phys.max())
ax1.set_ylim(y_phys.min(), y_phys.max())
ax1.axis('off')

plt.tight_layout()
plt.savefig('fig/trajectories_overlay.png', dpi=400, bbox_inches='tight', 
            facecolor='white', edgecolor='none', pad_inches=0)
plt.savefig('fig/trajectories_overlay.pdf', dpi=400, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0)
plt.close()

fig2, ax2 = plt.subplots(figsize=(10, 10))

nx, ny = x_phys.shape[0], y_phys.shape[0]
xi = np.linspace(x_phys.min(), x_phys.max(), nx)
yi = np.linspace(y_phys.min(), y_phys.max(), ny)
density = np.zeros((ny, nx))

sigma = 2.0  # Gaussian smoothing parameter
for trajectory in successful_trajectories:
    traj = np.array(trajectory)
    for point in traj:
        ix = (point[0] - x_phys.min()) / (x_phys.max() - x_phys.min()) * (nx-1)
        iy = (point[1] - y_phys.min()) / (y_phys.max() - y_phys.min()) * (ny-1)
        
        x_indices = np.arange(max(0, int(ix-3*sigma)), min(nx, int(ix+3*sigma)+1))
        y_indices = np.arange(max(0, int(iy-3*sigma)), min(ny, int(iy+3*sigma)+1))
        
        for xi_idx in x_indices:
            for yi_idx in y_indices:
                dist_sq = (xi_idx - ix)**2 + (yi_idx - iy)**2
                density[yi_idx, xi_idx] += 0.005 + np.exp(-dist_sq / (2 * sigma**2))

density_smooth = gaussian_filter(density, sigma=1.5)
density_transformed = np.log1p(density_smooth)
colors_custom = ['#FFFFFF','#E6F3FF', '#CCE7FF', '#80CCFF', '#3399FF', 
                '#0066CC', '#FFD700', '#FF8C00', '#FF4500', '#CC0000']
n_bins = 100
cmap_custom = LinearSegmentedColormap.from_list('trajectory_density', colors_custom, N=n_bins)

im = ax2.imshow(density_transformed, extent=[x_phys.min(), x_phys.max(), y_phys.min(), y_phys.max()],
                origin='lower', cmap=cmap_custom,  alpha=0.85, aspect='equal')

# Overlay capillary network
ax2.scatter(obstacle_contour[:, 0], obstacle_contour[:, 1], 
           color="black", s=0.2, alpha=0.6, rasterized=True)

ax2.set_aspect('equal')
ax2.set_xlim(x_phys.min(), x_phys.max())
ax2.set_ylim(y_phys.min(), y_phys.max())
ax2.axis('off')

plt.tight_layout()
plt.savefig('fig/trajectory_heatmap.png', dpi=400, bbox_inches='tight', 
            facecolor='white', edgecolor='none', pad_inches=0)
plt.savefig('fig/trajectory_heatmap.pdf', dpi=400, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0)
plt.close()

# =============================================================================
# Figure 3: Start/End points distribution
# =============================================================================
fig3, ax3 = plt.subplots(figsize=(10, 10))

# Calculate trajectory statistics
start_points = np.array([traj[0] for traj in successful_trajectories])
end_points = np.array([traj[-1] for traj in successful_trajectories])

# Overlay capillary network first (background)
ax3.scatter(obstacle_contour[:, 0], obstacle_contour[:, 1], 
           color="#2C2C2C", s=0.3, alpha=0.4, rasterized=True)

# Start points
ax3.scatter(start_points[:, 0], start_points[:, 1], 
           c='#00FF7F', s=30, alpha=0.8, marker='o', 
           edgecolors='white', linewidth=0.8, rasterized=True)

# End points
ax3.scatter(end_points[:, 0], end_points[:, 1], 
           c='#FF4500', s=30, alpha=0.8, marker='s',
           edgecolors='white', linewidth=0.8, rasterized=True)

ax3.set_aspect('equal')
ax3.set_xlim(x_phys.min(), x_phys.max())
ax3.set_ylim(y_phys.min(), y_phys.max())
ax3.axis('off')

plt.tight_layout()
plt.savefig('fig/start_end_distribution.png', dpi=400, bbox_inches='tight', 
            facecolor='white', edgecolor='none', pad_inches=0)
plt.savefig('fig/start_end_distribution.pdf', dpi=400, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0)
plt.close()

# =============================================================================
# Bonus: Create a colorbar separately for the heatmap
# =============================================================================
fig_cbar, ax_cbar = plt.subplots(figsize=(1, 8))
ax_cbar.axis('off')

# Create a dummy mappable for the colorbar
from matplotlib.cm import ScalarMappable
sm = ScalarMappable(cmap=cmap_custom)
sm.set_array(density_smooth)

cbar = plt.colorbar(sm, ax=ax_cbar, orientation='vertical', fraction=1.0, aspect=20)
cbar.set_label('Passage Frequency', fontsize=14, fontweight='bold')
cbar.ax.tick_params(labelsize=12)

plt.tight_layout()
plt.savefig('fig/heatmap_colorbar.png', dpi=400, bbox_inches='tight', 
            facecolor='white', edgecolor='none', pad_inches=0)
plt.savefig('fig/heatmap_colorbar.pdf', dpi=400, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0)
plt.close()

print("\n" + "="*60)
print("FIGURES SAVED SEPARATELY")
print("="*60)
print("1. trajectories_overlay.png/pdf - Raw trajectories with transparency")
print("2. trajectory_heatmap.png/pdf - High-resolution density heatmap")
print("3. start_end_distribution.png/pdf - Spatial distribution of tasks")
print("4. heatmap_colorbar.png/pdf - Colorbar for heatmap")
print("\nAll figures saved without axes or grids")
print("Resolution: 400 DPI for publication quality")
print("="*60)