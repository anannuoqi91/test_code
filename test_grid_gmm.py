#!/usr/bin/env python3
"""
Grid-Based Iterative GMM Clustering
Based on gmm.py, implements density grid clustering with GMM models
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import json
import glob
import os
import copy
from collections import defaultdict
from gmm_models_use import load_all_gmm_models as load_models
from gmm_models_use import load_grid_templates
from global_info import *


def select_gmm_for_grid_center(
    grid_center,
    grid_templates,
    grid_size,
    grid_bounds,
    use_median=True,
):
    # 1) 根据全局网格索引找标准模板
    row, col = grid_to_bin_idx(grid_center, grid_size, grid_bounds)
    tpl_list = grid_templates.get((row, col), None)
    if not tpl_list:
        return None, None

    for tpl in tpl_list:
        if tpl["covariance_type"] == "template":
            mean_vec = np.array(tpl["mean"], dtype=float)
            distance = np.linalg.norm(grid_center - mean_vec)
            best_candidate = copy.deepcopy(tpl)
            best_candidate["covariance"] = best_candidate["covariance"]["median"] if use_median else best_candidate["covariance"]["avg"]
            return best_candidate, distance
    return find_nearest_gmm(grid_center, tpl_list)


def get_points_center(points):
    """
    Calculate the center of a set of points

    Parameters:
        points: Array of points [N, 2]

    Returns:
        center: Center point [x, y]
    """
    return np.mean(points, axis=0)


def create_density_grid(points, grid_size=0.5, descending=True):
    x_min = points[:, 0].min()
    y_min = points[:, 1].min()

    # Count points in each grid cell
    grid_counts = defaultdict(int)

    for i, point in enumerate(points):
        x_idx = int((point[0] - x_min) / grid_size)
        y_idx = int((point[1] - y_min) / grid_size)
        grid_key = (x_idx, y_idx)
        grid_counts[grid_key] += 1
    # Sort grid cells by density (descending order)
    sorted_cells = sorted(grid_counts.items(),
                          key=lambda x: x[1],
                          reverse=descending)

    return sorted_cells


def load_all_gmm_models(data_dir):
    """
    Load all GMM models from JSON files in the data directory

    Returns:
        all_models: List of dictionaries with GMM parameters
    """
    json_pattern = os.path.join(data_dir, "**/gmm_models.json")
    json_files = glob.glob(json_pattern, recursive=True)

    if len(json_files) == 0:
        print(f"No gmm_models.json files found in {data_dir}")
        return []

    return load_models(json_files)


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

        # Extract [Y, Z] coordinates (map to X, Y)
        points_2d = np.column_stack([points_3d[:, 1], points_3d[:, 2]])

        # Data cleaning
        valid_mask = np.isfinite(points_2d).all(axis=1)
        points_2d = points_2d[valid_mask]

        coord_threshold = 1000
        valid_mask = (np.abs(points_2d) < coord_threshold).all(axis=1)
        points_2d = points_2d[valid_mask]

        if len(points_2d) < 3:
            return None

        return points_2d

    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None


def find_nearest_gmm(center, gmm_models):
    """
    Find the nearest GMM model to the given center point

    Parameters:
        center: Center point [x, y]
        gmm_models: List of GMM model dictionaries

    Returns:
        nearest_gmm: Nearest GMM model
        distance: Distance to nearest GMM
    """
    if len(gmm_models) == 0:
        return None, np.inf

    # Extract all GMM means
    means = np.array([model['mean'] for model in gmm_models])

    # Calculate distances
    distances = np.linalg.norm(means - center, axis=1)

    # Find nearest
    nearest_idx = np.argmin(distances)

    return gmm_models[nearest_idx], distances[nearest_idx]


def compute_nms_iou(cluster1_points, cluster2_points):
    """
    Compute IoU (Intersection over Union) between two clusters
    Used for Non-Maximum Suppression
    """
    # Convert to sets of point indices for efficient intersection
    set1 = set(map(tuple, cluster1_points))
    set2 = set(map(tuple, cluster2_points))

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0.0

    return intersection / union


def points_tolerance(gmm_n_points, n_assigned):
    if isinstance(gmm_n_points, dict):
        all_ratios = [n_assigned / gmm_n_points[key]
                      for key in gmm_n_points]
        return min(all_ratios)
    else:
        return n_assigned / gmm_n_points


def clustering_points(remaining_points, cov, gmm_n_points, grid_center, distance, min_cluster_size=5, n_points_tolerance=0.5):
    cov_inv = np.linalg.inv(cov)
    centered = remaining_points - grid_center
    mahal_distances = np.sqrt(
        np.sum((centered @ cov_inv) * centered, axis=1))
    assigned_mask = mahal_distances < sigma_threshold
    n_assigned = np.sum(assigned_mask)
    if n_assigned < min_cluster_size:
        print(f"    Too few points ({n_assigned}), skipping")
        return None
    # Calculate metrics
    inlier_ratio = n_assigned / len(remaining_points)
    mean_mahal = np.mean(mahal_distances[assigned_mask])
    # Validation: Check cluster size against GMM's n_points
    n_points_ratio = points_tolerance(gmm_n_points, n_assigned)
    n_points_deviation = abs(n_points_ratio - 1.0)
    is_reasonable = n_points_deviation <= n_points_tolerance
    # Calculate quality score (lower is better)
    # Combine: Mahalanobis distance + n_points deviation + distance to GMM
    quality_score = mean_mahal + n_points_deviation + distance * 0.1
    print(f"    Assigned: {n_assigned} points, Ratio: {n_points_ratio:.2f}, "
          f"Quality: {quality_score:.3f}, Status: {'✓' if is_reasonable else '✗'}")
    return {
        'assigned_mask': assigned_mask,
        'n_assigned': n_assigned,
        'inlier_ratio': inlier_ratio,
        'mean_mahal': mean_mahal,
        'n_points_ratio': n_points_ratio,
        'n_points_deviation': n_points_deviation,
        'is_reasonable': is_reasonable,
        'quality_score': quality_score,
        'gmm_n_points': gmm_n_points,
    }


def grid_based_iterative_clustering(points_2d, grid_templates, **kwargs):
    """
    Grid-based iterative clustering using GMM models with NMS

    Algorithm (IMPROVED):
    1. Find top-k potential cluster centers (high-density grids)
    2. For each grid center, perform GMM clustering
    3. Evaluate all candidate clusters and apply NMS
    4. Keep best cluster, remove assigned points
    5. Iterate on remaining points

    Parameters:
        points_2d: Input point cloud [N, 2]
        grid_templates: Dict[(row, col)] -> list of template models
        grid_size: Size of grid cells (meters)
        density_threshold: Minimum density ratio to continue (0-1)
        min_cluster_size: Minimum points for valid cluster
        max_iterations: Maximum number of clusters to find
        sigma_threshold: Sigma threshold for point assignment
        n_points_tolerance: Tolerance ratio for n_points validation (0-1)
                           e.g., 0.5 means cluster size should be within 50% of GMM's n_points

    Returns:
        clusters: List of point arrays for each cluster
        cluster_gmms: List of GMM models for each cluster
        cluster_centers: List of grid centers for each cluster
        remaining_points: Unassigned points
    """
    grid_size = kwargs.get('grid_size', 0.3)
    density_threshold = kwargs.get('density_threshold', 0.1)
    min_cluster_size = kwargs.get('min_cluster_size', 5)
    max_iterations = kwargs.get('max_iterations', 10)
    sigma_threshold = kwargs.get('sigma_threshold', 3.0)
    n_points_tolerance = kwargs.get('n_points_tolerance', 0.5)
    top_k_grids = kwargs.get('top_k_grids', 5)
    nms_iou_threshold = kwargs.get('nms_iou_threshold', 0.3)
    model_grid_size = kwargs.get('model_grid_size', 5.0)
    model_grid_bounds = kwargs.get('model_grid_bounds', (0.0, 0.0))
    print(f"\n{'='*60}")
    print("Grid-Based Iterative GMM Clustering")
    print(f"{'='*60}")
    print(f"Point cloud size: {len(points_2d)}")
    print(f"GMM grid cells available: {len(grid_templates)}")
    print(f"GMM grid cells size: {model_grid_size} m")
    print(f"GMM grid cells bounds: {model_grid_bounds} m")
    print(f"Grid size: {grid_size}m")
    print(f"Density threshold: {density_threshold*100:.1f}%")
    print(f"Sigma threshold: {sigma_threshold}σ")

    # Initial range check
    print(f"\nInitial range check...")
    remaining_points = points_2d.copy()
    remaining_indices = np.arange(len(points_2d))

    clusters = []
    cluster_gmms = []
    cluster_centers = []
    cluster_info = []

    # for iteration in range(max_iterations):
    iteration = 0
    while iteration < max_iterations and len(remaining_points) >= min_cluster_size:
        iteration += 1

        print(f"\n{'='*60}")
        print(
            f"Iteration {iteration+1}: {len(remaining_points)} remaining points")
        print(f"{'='*60}")

        # Step 1: Create density grid
        print(f"Step 1: Creating density grid...")
        x_min, x_max = remaining_points[:,
                                        0].min(), remaining_points[:, 0].max()
        y_min, y_max = remaining_points[:,
                                        1].min(), remaining_points[:, 1].max()

        # Count points in each grid cell
        sorted_cells = create_density_grid(
            remaining_points, grid_size=grid_size, descending=True)
        if len(sorted_cells) == 0:
            break
        # Step 2: Find top-k densest grid cells (potential cluster centers)
        print(f"Step 2: Finding top-{top_k_grids} densest grid cells...")
        top_cells = sorted_cells[:min(top_k_grids, len(sorted_cells))]

        # Step 3: For each grid center, perform GMM clustering
        print(f"\nStep 3: Performing GMM clustering on each grid center...")
        candidate_clusters = []
        for rank, (cell, count) in enumerate(top_cells, 1):
            density_ratio = count / len(remaining_points)
            print(
                f"    #{rank}: Cell {cell}, density: {count}/{len(remaining_points)} ({density_ratio*100:.1f}%)")

            # Calculate grid cell center
            grid_center = grid_idx_to_point(cell, grid_size, (x_min, y_min))

            print(f"\n  Candidate #{rank}:")
            print(
                f"    Grid center: [{grid_center[0]:.3f}, {grid_center[1]:.3f}]")

            nearest_gmm, distance = select_gmm_for_grid_center(
                grid_center, grid_templates, model_grid_size, model_grid_bounds)
            # Find nearest GMM to this grid center
            # nearest_gmm, distance = find_nearest_gmm(grid_center, gmm_models)

            if nearest_gmm is None:
                print(f"    No GMM found, skipping")
                continue

            print(f"    Nearest GMM: "
                  f"Mean=[{nearest_gmm['mean'][0]:.3f}, {nearest_gmm['mean'][1]:.3f}], "
                  f"n_points={nearest_gmm['n_points']}, Distance={distance:.3f}m")

            # Perform GMM clustering
            cov = nearest_gmm['covariance']

            try:
                tmp = clustering_points(
                    remaining_points, cov, nearest_gmm["n_points"], grid_center, distance, min_cluster_size, n_points_tolerance)
                if tmp is None:
                    continue

                print(f"    Assigned: {tmp['n_assigned']} points, Ratio: {tmp['n_points_ratio']:.2f}, "
                      f"Quality: {tmp['quality_score']:.3f}, Status: {'✓' if tmp['is_reasonable'] else '✗'}")

                # Store candidate cluster
                cluster_points = remaining_points[tmp['assigned_mask']].copy()
                cluster_point_indices = remaining_indices[tmp['assigned_mask']].copy(
                )
                info = {
                    'grid_center': grid_center,
                    'gmm': nearest_gmm,
                    'distance_to_gmm': distance,
                    'points': cluster_points,
                    'point_indices': cluster_point_indices,
                    'rank': rank
                }
                info.update(tmp)
                candidate_clusters.append(info)

            except np.linalg.LinAlgError:
                print(f"    Singular covariance matrix, skipping")
                continue

        # Step 4: Evaluate candidates and apply NMS
        print(
            f"\nStep 4: Evaluating {len(candidate_clusters)} candidate clusters and applying NMS...")

        if len(candidate_clusters) == 0:
            print(f"  No valid candidates found, stopping")
            break

        # Sort candidates by quality score (lower is better)
        candidate_clusters.sort(key=lambda x: x['quality_score'])

        # Apply NMS
        selected_clusters = []
        for i, candidate in enumerate(candidate_clusters):
            # Check IoU with already selected clusters
            should_keep = True
            for selected in selected_clusters:
                iou = compute_nms_iou(candidate['points'], selected['points'])
                if iou > nms_iou_threshold:
                    print(
                        f"  Candidate #{candidate['rank']} suppressed by NMS (IoU={iou:.2f} with selected cluster)")
                    should_keep = False
                    break

            if should_keep:
                selected_clusters.append(candidate)
                print(
                    f"  ✓ Candidate #{candidate['rank']} selected (Quality: {candidate['quality_score']:.3f})")

        if len(selected_clusters) == 0:
            print(f"  No clusters passed NMS, stopping")
            break

        # Keep only the best cluster from this iteration
        best_candidate = selected_clusters[0]

        print(f"\n  Best cluster:")
        print(
            f"    Grid center: [{best_candidate['grid_center'][0]:.3f}, {best_candidate['grid_center'][1]:.3f}]")
        print(f"    Points: {best_candidate['n_assigned']}")
        print(f"    Quality: {best_candidate['quality_score']:.3f}")
        print(
            f"    Status: {'REASONABLE ✓' if best_candidate['is_reasonable'] else 'UNREASONABLE ✗'}")

        # Check if we should stop based on n_points validation
        if not best_candidate['is_reasonable']:
            if isinstance(best_candidate["gmm_n_points"], dict):
                min_points = best_candidate["gmm_n_points"]["min"]
            else:
                min_points = best_candidate["gmm_n_points"]

            if best_candidate['n_assigned'] < min_points * (1 - n_points_tolerance):
                print(
                    f"  ⚠️  Too few points compared to GMM model, remaining points likely noise")
                print(f"     Stopping iteration")
                break

        # Store the best cluster
        cluster_points = best_candidate['points']
        cluster_point_indices = best_candidate['point_indices']

        clusters.append(cluster_points)
        cluster_gmms.append(best_candidate['gmm'])
        cluster_centers.append(best_candidate['grid_center'])

        cluster_info.append({
            'cluster_id': len(clusters),
            'n_points': best_candidate['n_assigned'],
            'gmm_n_points': best_candidate['gmm_n_points'],
            'n_points_ratio': best_candidate['n_points_ratio'],
            'n_points_deviation': best_candidate['n_points_deviation'],
            'is_reasonable': best_candidate['is_reasonable'],
            'grid_center': best_candidate['grid_center'],
            'gmm_mean': best_candidate['gmm']['mean'],
            'distance_to_gmm': best_candidate['distance_to_gmm'],
            'inlier_ratio': best_candidate['inlier_ratio'],
            'mean_mahal': best_candidate['mean_mahal']
        })

        actual_center = np.mean(cluster_points, axis=0)
        print(f"\n  Stored Cluster {len(clusters)}:")
        print(f"    Points: {best_candidate['n_assigned']}")
        print(
            f"    Actual center: [{actual_center[0]:.3f}, {actual_center[1]:.3f}]")
        print(f"    Inlier ratio: {best_candidate['inlier_ratio']*100:.1f}%")
        print(f"    Mean Mahalanobis: {best_candidate['mean_mahal']:.3f}")

        # Step 5: Remove assigned points
        remaining_points = remaining_points[~best_candidate['assigned_mask']]
        remaining_indices = remaining_indices[~best_candidate['assigned_mask']]

    if len(remaining_points) < min_cluster_size:
        print(
            f"\nIteration {iteration+1}: Insufficient remaining points ({len(remaining_points)})")

    n_targets = len(clusters)

    print(f"\n{'='*60}")
    print(f"Clustering Complete: {n_targets} target(s) detected")
    print(f"{'='*60}")

    for info in cluster_info:
        status_icon = '✓' if info['is_reasonable'] else '✗'
        print(
            f"  Target {info['cluster_id']}: {info['n_points']} points {status_icon}")
        print(
            f"    Grid center: [{info['grid_center'][0]:.3f}, {info['grid_center'][1]:.3f}]")
        print(f"    Distance to GMM: {info['distance_to_gmm']:.3f}m")
        print(f"    Inlier ratio: {info['inlier_ratio']*100:.1f}%")
        print(f"    n_points validation: {info['n_points']}/{info['gmm_n_points']} "
              f"(ratio: {info['n_points_ratio']:.2f}, deviation: {info['n_points_deviation']*100:.1f}%)")
        print(
            f"    Status: {'REASONABLE' if info['is_reasonable'] else 'UNREASONABLE'}")

    if len(remaining_points) > 0:
        print(f"  Unassigned: {len(remaining_points)} points")

    return clusters, cluster_gmms, cluster_centers, remaining_points, cluster_info


def visualize_clustering_result(points_2d, clusters, cluster_gmms, cluster_centers,
                                remaining_points, cluster_info, output_file=None):
    """
    Visualize the clustering results
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))

    # Plot each cluster
    for i, (cluster, gmm, center, color) in enumerate(zip(clusters, cluster_gmms, cluster_centers, colors)):
        # Check if cluster is reasonable
        info = cluster_info[i]
        status = '✓' if info['is_reasonable'] else '✗'

        # Plot cluster points
        ax.scatter(cluster[:, 0], cluster[:, 1], c=[color], s=50, alpha=0.6,
                   label=f"Cluster {i+1} ({len(cluster)} pts) {status}", edgecolors='black', linewidth=0.5)

        # Plot grid center
        ax.scatter(center[0], center[1], c=[color], marker='X', s=300,
                   edgecolors='black', linewidth=2, zorder=10)

        # Plot GMM mean
        gmm_mean = gmm['mean']
        ax.scatter(gmm_mean[0], gmm_mean[1], c=[color], marker='s', s=200,
                   edgecolors='black', linewidth=2, alpha=0.7, zorder=9)

        # Draw covariance ellipses
        cov = gmm['covariance']
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        for n_std in [1, 2, 3]:
            width, height = 2 * n_std * np.sqrt(eigenvalues)
            ellipse = Ellipse(center, width, height, angle=angle,
                              facecolor='none', edgecolor=color,
                              linewidth=2, linestyle='--', alpha=0.5)
            ax.add_patch(ellipse)

    # Plot remaining points
    if len(remaining_points) > 0:
        ax.scatter(remaining_points[:, 0], remaining_points[:, 1],
                   c='gray', s=30, alpha=0.3, label=f"Unassigned ({len(remaining_points)} pts)")

    ax.set_xlabel('X (original Y) [m]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (original Z) [m]', fontsize=12, fontweight='bold')
    ax.set_title('Grid-Based Iterative GMM Clustering Result',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal', adjustable='datalim')

    # Add statistics text
    stats_text = f"Total points: {len(points_2d)}\n"
    stats_text += f"Clusters: {len(clusters)}\n"
    stats_text += f"Assigned: {sum(len(c) for c in clusters)}\n"
    stats_text += f"Unassigned: {len(remaining_points)}"

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            family='monospace')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_file}")

    plt.show()


if __name__ == "__main__":
    # Configuration
    data_dir = "./data"
    pcd_file = "./data/two_dense3.pcd"
    base_name = os.path.basename(pcd_file).split(".")[0]
    output_file = f"./result/grid_gmm_result_{base_name}.png"

    # Parameters
    grid_size = 0.3           # Grid cell size (meters)
    density_threshold = 0.1   # Minimum density ratio (10%)
    min_cluster_size = 5      # Minimum points per cluster
    max_iterations = 100       # Maximum number of clusters
    sigma_threshold = 3.0     # Sigma threshold for point assignment
    n_points_tolerance = 0.5  # Tolerance for n_points validation (50%)
    top_k_grids = 5           # Number of candidate grid centers to evaluate
    nms_iou_threshold = 0.3   # IoU threshold for NMS (30%)

    # Load GMM models
    print("Loading GMM models...")
    models_dir = './data_/single_lidar'
    models_dir = os.path.join(models_dir, "grid_models.json")
    grid_templates, model_grid_size, model_grid_bounds = load_grid_templates(
        models_dir)

    # Read point cloud
    print(f"\nReading point cloud: {pcd_file}")
    points_2d = read_pcd_and_extract_2d(pcd_file)

    if points_2d is None:
        print("Failed to read point cloud")
        exit(1)

    # Perform clustering
    clusters, cluster_gmms, cluster_centers, remaining_points, cluster_info = \
        grid_based_iterative_clustering(points_2d, grid_templates,
                                        grid_size=grid_size,
                                        density_threshold=density_threshold,
                                        min_cluster_size=min_cluster_size,
                                        max_iterations=max_iterations,
                                        sigma_threshold=sigma_threshold,
                                        n_points_tolerance=n_points_tolerance,
                                        top_k_grids=top_k_grids,
                                        nms_iou_threshold=nms_iou_threshold, model_grid_size=model_grid_size, model_grid_bounds=model_grid_bounds)

    # Visualize results
    visualize_clustering_result(points_2d, clusters, cluster_gmms, cluster_centers,
                                remaining_points, cluster_info, output_file)

    print("\nClustering complete!")
