#!/usr/bin/env python3
"""
Train GMM models for multiple PCD files
Reads all PCD files from a directory, fits GMM for each, and saves the parameters
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os
import glob
import json
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from global_info import *


""""
离线模型
分网格比对聚类参数的一致性
"""


def read_pcd_and_extract_2d(filename):
    """
    Read PCD file and extract 2D coordinates [Y, Z]
    Returns:
        points_2d: 2D point cloud [Y, Z] or None if failed
    """
    try:
        pcd = o3d.io.read_point_cloud(filename)
        points_3d = np.asarray(pcd.points)
        if points_3d.shape[0] == 0 or points_3d.shape[1] < 3:
            return None
        # Extract [Y, Z] coordinates
        points_2d = np.column_stack([points_3d[:, 1], points_3d[:, 2]])
        # Data cleaning
        valid_mask = np.isfinite(points_2d).all(axis=1)
        points_2d = points_2d[valid_mask]
        coord_threshold = 1000
        valid_mask = (np.abs(points_2d) < coord_threshold).all(axis=1)
        points_2d = points_2d[valid_mask]
        if len(points_2d) < 5:  # Need at least 5 points for GMM
            return None
        return points_2d
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None


def fit_gmm_single_file(points_2d, covariance_type='diag'):
    """
    Fit GMM for a single point cloud

    Returns:
        params: Dictionary with GMM parameters or None if failed
    """
    try:
        gmm = GaussianMixture(
            n_components=1,
            covariance_type=covariance_type,
            random_state=42,
            max_iter=200,
            n_init=10
        )

        gmm.fit(points_2d)

        mean = gmm.means_[0]

        if covariance_type == 'full':
            cov = gmm.covariances_[0]
        elif covariance_type == 'diag':
            cov = np.diag(gmm.covariances_[0])
        elif covariance_type == 'spherical':
            cov = np.eye(2) * gmm.covariances_[0]

        params = {
            'mean': mean.tolist(),
            'covariance': cov.tolist(),
            'covariance_type': covariance_type,
            'n_points': len(points_2d),
            'converged': bool(gmm.converged_),
            'n_iter': int(gmm.n_iter_),
            'log_likelihood': float(gmm.lower_bound_),
            'aic': float(gmm.aic(points_2d)),
            'bic': float(gmm.bic(points_2d))
        }

        return params
    except Exception as e:
        print(f"Error fitting GMM: {e}")
        return None


def train_gmm_batch(data_dir, output_file, covariance_type='diag'):
    """
    Train GMM models for all PCD files in a directory

    Parameters:
        data_dir: Directory containing PCD files
        output_file: JSON file to save GMM parameters
        covariance_type: Type of covariance ('full', 'diag', 'spherical')

    Returns:
        gmm_models: Dictionary mapping filename to GMM parameters
    """
    # Find all PCD files
    pcd_files = sorted(glob.glob(os.path.join(data_dir, "*.pcd")))

    if len(pcd_files) == 0:
        print(f"No PCD files found in {data_dir}")
        return {}

    print(f"Found {len(pcd_files)} PCD files")
    print(f"Training GMM models with covariance_type='{covariance_type}'...")

    gmm_models = {}
    failed_files = []

    # Process each file with progress bar
    for pcd_file in tqdm(pcd_files, desc="Training GMMs"):
        filename = os.path.basename(pcd_file)

        # Read and extract 2D points
        points_2d = read_pcd_and_extract_2d(pcd_file)

        if points_2d is None:
            failed_files.append(filename)
            continue

        # Fit GMM
        params = fit_gmm_single_file(points_2d, covariance_type)

        if params is None:
            failed_files.append(filename)
            continue

        # Store parameters
        gmm_models[filename] = params

    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(gmm_models, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"{'='*60}")
    print(f"Successfully trained: {len(gmm_models)} models")
    print(f"Failed: {len(failed_files)} files")
    if failed_files:
        print(
            f"Failed files: {failed_files[:5]}{'...' if len(failed_files) > 5 else ''}")
    print(f"Models saved to: {output_file}")
    print(f"{'='*60}\n")

    return gmm_models


def visualize_gmm_coverage(gmm_models, output_image=None):
    """
    Visualize the spatial coverage of trained GMM models

    Parameters:
        gmm_models: Dictionary of GMM parameters
        output_image: Path to save the visualization (optional)
    """
    if len(gmm_models) == 0:
        print("No models to visualize")
        return

    # Extract all means and covariances
    means = []
    covariances = []
    filenames = []

    for filename, params in gmm_models.items():
        means.append(params['mean'])
        covariances.append(params['covariance'])
        filenames.append(filename)

    means = np.array(means)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    # Left plot: All GMM centers with trajectory
    ax1.plot(means[:, 0], means[:, 1], 'b-', alpha=0.3,
             linewidth=1, label='Trajectory')
    ax1.scatter(means[:, 0], means[:, 1], c=range(len(means)), cmap='viridis',
                s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

    # Mark start and end
    ax1.scatter(means[0, 0], means[0, 1], c='green', marker='o', s=200,
                edgecolors='black', linewidth=2, label='Start', zorder=10)
    ax1.scatter(means[-1, 0], means[-1, 1], c='red', marker='s', s=200,
                edgecolors='black', linewidth=2, label='End', zorder=10)

    ax1.set_xlabel('X (original Y coordinate)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Y (original Z coordinate)', fontsize=14, fontweight='bold')
    ax1.set_title(f'GMM Model Coverage - Trajectory ({len(gmm_models)} models)',
                  fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_aspect('equal', adjustable='datalim')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(vmin=0, vmax=len(means)-1))
    sm.set_array([])
    cbar1 = plt.colorbar(sm, ax=ax1)
    cbar1.set_label('Frame Index', fontsize=12, fontweight='bold')

    # Right plot: GMM ellipses (sample every N frames to avoid clutter)
    sample_rate = max(1, len(gmm_models) // 20)  # Show at most 20 ellipses
    sampled_indices = range(0, len(means), sample_rate)

    for idx in sampled_indices:
        mean = means[idx]
        cov = np.array(covariances[idx])

        # Plot center
        color = plt.cm.viridis(idx / len(means))
        ax2.scatter(mean[0], mean[1], c=[color], s=30,
                    edgecolors='black', linewidth=0.5, zorder=5)

        # Draw 2σ ellipse
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * 2 * np.sqrt(eigenvalues)  # 2σ

        ellipse = Ellipse(mean, width, height, angle=angle,
                          facecolor='none', edgecolor=color,
                          linewidth=1.5, linestyle='--', alpha=0.6)
        ax2.add_patch(ellipse)

    # Mark start and end
    ax2.scatter(means[0, 0], means[0, 1], c='green', marker='o', s=200,
                edgecolors='black', linewidth=2, label='Start', zorder=10)
    ax2.scatter(means[-1, 0], means[-1, 1], c='red', marker='s', s=200,
                edgecolors='black', linewidth=2, label='End', zorder=10)

    ax2.set_xlabel('X (original Y coordinate)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Y (original Z coordinate)', fontsize=14, fontweight='bold')
    ax2.set_title(f'GMM Model Coverage - 2σ Ellipses (sampled every {sample_rate} frames)',
                  fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_aspect('equal', adjustable='datalim')

    # Statistics text
    stats_text = f"Total Models: {len(gmm_models)}\n"
    stats_text += f"X range: [{means[:, 0].min():.2f}, {means[:, 0].max():.2f}]\n"
    stats_text += f"Y range: [{means[:, 1].min():.2f}, {means[:, 1].max():.2f}]\n"
    stats_text += f"Distance traveled: {np.sum(np.linalg.norm(np.diff(means, axis=0), axis=1)):.2f}m"

    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             family='monospace')

    plt.tight_layout()

    if output_image:
        plt.savefig(output_image, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_image}")

    # plt.show()


def print_summary_statistics(gmm_models):
    """Print summary statistics of trained models"""
    if len(gmm_models) == 0:
        return

    means = np.array([params['mean'] for params in gmm_models.values()])
    n_points = [params['n_points'] for params in gmm_models.values()]
    converged = [params['converged'] for params in gmm_models.values()]

    print(f"\n{'='*60}")
    print(f"Summary Statistics")
    print(f"{'='*60}")
    print(f"Total models: {len(gmm_models)}")
    print(
        f"Converged models: {sum(converged)} ({sum(converged)/len(converged)*100:.1f}%)")
    print(f"\nMean positions:")
    print(
        f"  X: min={means[:, 0].min():.3f}, max={means[:, 0].max():.3f}, avg={means[:, 0].mean():.3f}")
    print(
        f"  Y: min={means[:, 1].min():.3f}, max={means[:, 1].max():.3f}, avg={means[:, 1].mean():.3f}")
    print(f"\nPoints per model:")
    print(
        f"  min={min(n_points)}, max={max(n_points)}, avg={np.mean(n_points):.1f}")
    print(
        f"\nTrajectory length: {np.sum(np.linalg.norm(np.diff(means, axis=0), axis=1)):.2f}m")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Configuration
    data_dir = "./data/single_lidar/A12"
    for i in os.listdir(data_dir):
        if i.endswith(".json"):
            continue
        pcd_dir = os.path.join(data_dir, i)
        # Save in same directory as PCD files
        output_json = os.path.join(pcd_dir, "gmm_models.json")
        # Save in same directory as PCD files
        output_image = os.path.join(pcd_dir, "gmm_coverage.png")
        covariance_type = 'diag'  # Options: 'full', 'diag', 'spherical'

        # Train GMM models
        gmm_models = train_gmm_batch(pcd_dir, output_json, covariance_type)

        # Print statistics
        print_summary_statistics(gmm_models)

        # Visualize coverage
        visualize_gmm_coverage(gmm_models, output_image)
