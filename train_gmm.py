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


"""
模型训练
1.根据接入的vru数据进行模型训练
"""


class GMMTrainer(ModelScorer):
    def __init__(self, n_components=1,
                 max_iter=200,
                 n_init=10,
                 tol=1e-5,
                 covariance_type="diag"):
        super().__init__()
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.covariance_type = covariance_type
        self.n_init = n_init

    def train(self, points_2d):
        out = ResultTrainer()
        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=42,
            max_iter=self.max_iter,
            n_init=self.n_init,
            tol=self.tol
        )
        try:
            gmm.fit(points_2d)
            out.mean_point = gmm.means_[0].tolist()
            if self.covariance_type == 'full':
                cov = gmm.covariances_[0]
            elif self.covariance_type == 'diag':
                cov = np.diag(gmm.covariances_[0])
            elif self.covariance_type == 'spherical':
                cov = np.eye(2) * gmm.covariances_[0]
            out.covariance = cov.tolist()
            out.covariance_type = self.covariance_type
            out.n_points = len(points_2d)
            out.converged = bool(gmm.converged_)
            out.n_iter = int(gmm.n_iter_)
            out.log_likelihood = float(gmm.lower_bound_)
            out.aic = float(gmm.aic(points_2d))
            out.bic = float(gmm.bic(points_2d))
            out.update = True
            out.max_iter = self.max_iter
            out.self_score = float(
                self.compute_score(out, points_2d=points_2d))
        except Exception as e:
            print(f"Error training GMM: {e}")
        finally:
            return out

    def visualize_gmm(self, points_2d, gmm_result: ResultTrainer, output_file=None, is_show=False):
        """
        可视化 GMM 模型在 2D 点云上的拟合结果
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        colors = plt.cm.tab10(np.linspace(0, 1, 1))
        center = gmm_result.mean_point
        color = colors[0]
        # Plot cluster points
        ax.scatter(points_2d[:, 0], points_2d[:, 1],
                   c=[colors[0]], s=50, alpha=0.6,
                   label=f"Cluster({len(points_2d)} pts)", edgecolors='black', linewidth=0.5)
        # Plot grid center
        ax.scatter(center[0], center[1], c=[color], marker='X', s=300,
                   edgecolors='black', linewidth=2, zorder=10)
        # Draw covariance ellipses
        cov = gmm_result.covariance
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        for n_std in [1, 2, 3]:
            width, height = 2 * n_std * np.sqrt(eigenvalues)
            ellipse = Ellipse(center, width, height, angle=angle,
                              facecolor='none', edgecolor=color,
                              linewidth=2, linestyle='--', alpha=0.5)
            ax.add_patch(ellipse)
        ax.set_xlabel('X (original Y) [m]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (original Z) [m]', fontsize=12, fontweight='bold')
        ax.set_title(f'GMM Clustering Result (Score: {gmm_result.self_score:.3f})',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='datalim')

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to: {output_file}")
        if is_show:
            plt.show()
        plt.close(fig)


def train_gmm_batch(data_dir, output_file, covariance_type='diag',
                    png_dir=None):
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
    gmm_trainer = GMMTrainer(
        n_components=1,
        max_iter=200,
        n_init=10,
        tol=1e-5,
        covariance_type=covariance_type
    )

    # Process each file with progress bar
    for pcd_file in tqdm(pcd_files, desc="Training GMMs"):
        filename = os.path.basename(pcd_file)

        # Read and extract 2D points
        points_2d = read_pcd_and_extract_2d(pcd_file)

        if points_2d is None:
            failed_files.append(filename)
            continue

        # Fit GMM
        params = gmm_trainer.train(points_2d)

        if params is None:
            failed_files.append(filename)
            continue
        if png_dir:
            png_file = os.path.join(png_dir, filename.replace(".pcd", ".png"))
            gmm_trainer.visualize_gmm(
                points_2d, params, png_file, is_show=False)

        # Store parameters
        gmm_models[filename] = params.to_dict()

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
        means.append(params['mean_point'])
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

    means = np.array([params['mean_point'] for params in gmm_models.values()])
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
    data_dir = "./data/VRU_Passing_B36_002_FK_0_0/train"
    png_dir = "./result/VRU_Passing_B36_002_FK_0_0/png/train"
    for i in os.listdir(data_dir):
        if i.endswith(".json"):
            continue
        pcd_dir = os.path.join(data_dir, i)
        # Save in same directory as PCD files
        output_json = os.path.join(pcd_dir, "gmm_models.json")
        # Save in same directory as PCD files
        output_image = os.path.join(pcd_dir, "gmm_coverage.png")
        covariance_type = 'diag'  # Options: 'full', 'diag', 'spherical'
        i_png_dir = os.path.join(png_dir, i)
        os.makedirs(i_png_dir, exist_ok=True)

        # Train GMM models
        gmm_models = train_gmm_batch(
            pcd_dir, output_json, covariance_type, png_dir=i_png_dir)

        # Print statistics
        print_summary_statistics(gmm_models)

        # Visualize coverage
        visualize_gmm_coverage(gmm_models, output_image)
