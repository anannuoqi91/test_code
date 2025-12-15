from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import json
from global_info import *


class PredictGMM:
    def __init__(self, model_grid_info: ModelGridInfo,
                 models: Dict[Tuple[int, int], List[GridGMMModel]],
                 min_cluster_points: int = 5,
                 score_thresh: float = None, **kwargs):
        self.model_grid_size = model_grid_info.grid_size
        self.model_grid_bounds = (
            model_grid_info.origin_y, model_grid_info.origin_z)
        self.models = models
        self.min_cluster_points = min_cluster_points
        self.score_thresh = score_thresh
        self.grid_size = kwargs.get("grid_size", 0.3)
        self.density_threshold = kwargs.get("density_threshold", 0.1)
        self.sigma_threshold = kwargs.get("sigma_threshold", 3.0)
        self.n_points_tolerance = kwargs.get("n_points_tolerance", 0.5)
        self.top_k_grids = kwargs.get("top_k_grids", 5)
        self.nms_iou_threshold = kwargs.get("nms_iou_threshold", 0.3)
        self.debug = kwargs.get("debug", False)
        self.max_iterations = kwargs.get("max_iterations", 10)
        self.model_use_median = kwargs.get("model_use_median", True)
        self.mahal_threshold = kwargs.get("mahal_threshold", 2.0)
        self.inlier_ratio_min = kwargs.get("inlier_ratio_min", 0.5)
        self.init_ratio_min = kwargs.get("init_ratio_min", 0.15)
        self._model_found = False
        self._result_dic = None

    def _init_out(self, points_2d):
        return {
            "info": {
                "n_points": points_2d.shape[0],
                "model_found": False,
                "center_mean": np.mean(points_2d, axis=0),
                "points_2d": points_2d,
                "remaining_points": [[]]
            },
            "clusters": []}

    def _out_dic(self, **kwargs):
        return {
            "points": kwargs["cluster"],
            "points_num": len(kwargs["cluster"]),
            "gmm": kwargs["gmm"],
            "cluster_center": kwargs["cluster_center"],
            "quality_score": kwargs["quality_score"],
            "is_reasonable": kwargs["is_reasonable"],
            "distance_to_gmm": kwargs["distance_to_gmm"],
            "mean_mahal": kwargs["mean_mahal"],
            "inlier_ratio": kwargs["inlier_ratio"],
            "actual_center": np.mean(kwargs["cluster"], axis=0),
            "points_score_to_gmm": kwargs["points_score_to_gmm"]
        }

    def _print_log(self, msg: str):
        if self.debug:
            print(msg)

    def _cal_predict_quality_score(self, mean_mahal, n_points_deviation,
                                   dis_to_gmm, init_ratio):
        # quality_score ∈ [0,1]，越小越好
        # 1) Mahalanobis 平均距离：期望 <= mahal_threshold 是好，>= 2*mahal_threshold 认为很差
        pen_mahal = self._linear_penalty(
            mean_mahal,
            good=0.0,
            bad=self.mahal_threshold * 2,   # 可按经验调，比如 2~3 倍
        )
        # 2) 点数偏差：期望 0 最好，超过 n_points_tolerance 就逐渐视为不好
        pen_npts = self._linear_penalty(
            n_points_deviation,
            good=0.0,
            bad=1.0
        )

        # 3) 模板中心距离：0 最好，超过一定距离就当不好
        #    这里简单用 2 * grid_size 当“明显偏远”，可以按数据再调
        dist_good = 0.0
        dist_bad = self.grid_size
        pen_dist = self._linear_penalty(
            dis_to_gmm,
            good=dist_good,
            bad=dist_bad,
        )
        # 4) init_ratio：越大越好，这里反过来用 (1 - init_ratio) 做“惩罚”
        pen_init = self._linear_penalty(
            1.0 - init_ratio,
            good=0.0,                         # init_ratio = 1.0 -> 惩罚 0
            bad=1.0,    # init_ratio = init_ratio_min -> 差
        )
        # 5) 综合质量得分：加权平均
        # Mahalanobis：0.4（拟合几何的核心指标）
        # 点数偏差：0.3（数量是否匹配模板）
        # 距离惩罚：0.1（模板位置偏移）
        # init_ratio 惩罚：0.2（迭代阶段收敛的质量）
        w_mahal = 0.4
        w_npts = 0.3
        w_dist = 0.1
        w_init = 0.2

        raw_quality = (
            w_mahal * pen_mahal +
            w_npts * pen_npts +
            w_dist * pen_dist +
            w_init * pen_init
        )
        quality_score = float(np.clip(raw_quality, 0.0, 1.0))
        return quality_score

    def _create_density_grid(self, points_2d: np.ndarray, descending=True):
        x_min = points_2d[:, 0].min()
        y_min = points_2d[:, 1].min()

        # Count points in each grid cell
        grid_counts = defaultdict(int)

        for i, point in enumerate(points_2d):
            x_idx = int((point[0] - x_min) / self.grid_size)
            y_idx = int((point[1] - y_min) / self.grid_size)
            grid_key = (x_idx, y_idx)
            grid_counts[grid_key] += 1
        # Sort grid cells by density (descending order)
        sorted_cells = sorted(grid_counts.items(),
                              key=lambda x: x[1],
                              reverse=descending)
        return sorted_cells

    def _find_nearest_gmm(self, grid_center, tpl_list):
        min_distance = np.inf
        if len(tpl_list) == 0:
            return None, min_distance
        means = np.array([tpl.mean_point for tpl in tpl_list])
        distances = np.linalg.norm(means - grid_center, axis=1)
        nearest_idx = np.argmin(distances)
        return tpl_list[nearest_idx], distances[nearest_idx]

    def _model_points(self, tpl, use_preferred=True):
        if tpl.covariance_type in [CovarianceType.TEMPLATE, CovarianceType.BIN_TEMPLATE]:
            if use_preferred:
                return tpl.template_model.median_n_points if tpl.template_model.point_use_preferred == "median" else tpl.template_model.avg_n_points
            else:
                return tpl.template_model.median_n_points
        else:
            return tpl.n_points

    def _select_gmm_for_grid_center(self, grid_center):
        row, col = grid_to_bin_idx(
            grid_center, self.model_grid_size, self.model_grid_bounds)
        tpl_list = self.models.get((row, col), [])
        if not tpl_list:
            return None, None
        for tpl in tpl_list:
            if tpl.covariance_type == CovarianceType.TEMPLATE:
                mean_vec = np.array(tpl.mean_point, dtype=float)
                distance = np.linalg.norm(grid_center - mean_vec)
                useful_conv = tpl.template_model.covariance_median if self.model_use_median else tpl.template_model.covariance_avg
                n_points = self._model_points(tpl)
                return {"covariance": useful_conv,
                        "mean_point": tpl.mean_point,
                        "n_points": n_points,
                        "type": tpl.covariance_type.value,
                        "key": (row, col)}, distance
        tmp_tpl, distance = self._find_nearest_gmm(grid_center, tpl_list)
        if tmp_tpl.covariance_type == CovarianceType.BIN_TEMPLATE:
            useful_conv = tmp_tpl.template_model.covariance_median if self.model_use_median else tmp_tpl.template_model.covariance_avg
            n_points = self._model_points(tmp_tpl)
        else:
            useful_conv = tmp_tpl.covariance
            n_points = tmp_tpl.n_points
        return {"covariance": useful_conv,
                "mean_point": tmp_tpl.mean_point,
                "n_points": n_points,
                "type": tpl.covariance_type.value,
                "key": (row, col)}, distance

    def _points_tolerance(self, gmm_n_points, n_assigned):
        ratio = n_assigned / gmm_n_points
        ratio_abs = abs(ratio - 1.0)
        return ratio, ratio_abs

    def _linear_penalty(self, value, good, bad):
        """
        把一个“越大越差”的值 value 映射到 [0,1]：
        - value <= good   -> 0.0  (最好)
        - value >= bad    -> 1.0  (最差)
        - 中间线性插值
        """
        if value < good:
            return 0.0
        if value > bad:
            return 1.0
        return float((value - good) / (bad - good))

    def _clustering_points(self, gmm, remaining_points,
                           init_points_num, dis_to_gmm, grid_center):
        cov_inv = np.linalg.inv(gmm["covariance"])
        gmm_n_points = gmm["n_points"]
        centered = remaining_points - grid_center
        mahal_distances = np.sqrt(
            np.sum((centered @ cov_inv) * centered, axis=1))
        assigned_mask = mahal_distances < self.sigma_threshold
        n_assigned = np.sum(assigned_mask)
        if n_assigned < self.min_cluster_points:
            print(
                f"[clustering_points] Too few points ({n_assigned}), skipping")
            return None
        inlier_ratio = n_assigned / len(remaining_points)
        mean_mahal = np.mean(mahal_distances[assigned_mask])
        init_ratio = n_assigned / init_points_num
        good_fit = (mean_mahal <= self.mahal_threshold) and (
            inlier_ratio >= self.inlier_ratio_min) and (init_ratio >= self.init_ratio_min)
        n_points_ratio, n_points_deviation = self._points_tolerance(
            gmm_n_points, n_assigned)

        is_reasonable = (n_points_deviation <=
                         self.n_points_tolerance) or good_fit
        # quality_score = self._cal_predict_quality_score(
        #     mean_mahal, n_points_deviation, dis_to_gmm, init_ratio)
        quality_score = mean_mahal + n_points_deviation + \
            dis_to_gmm * 0.1 + (1 - init_ratio)

        self._print_log(f"    Assigned: {n_assigned} points, Ratio: {n_points_ratio:.2f}, "
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

    def _compute_nms_iou(self, cluster1_points, cluster2_points):
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

    def _evaluate_cluster(self, candidate_clusters):
        candidate_clusters.sort(key=lambda x: x['quality_score'])
        # Apply NMS
        selected_clusters = []
        for i, candidate in enumerate(candidate_clusters):
            # Check IoU with already selected clusters
            should_keep = True
            for selected in selected_clusters:
                iou = self._compute_nms_iou(
                    candidate['points'], selected['points'])
                if iou > self.nms_iou_threshold:
                    self._print_log(
                        f"  Candidate #{candidate['rank']} suppressed by NMS (IoU={iou:.2f} with selected cluster #{selected['rank']})")
                    should_keep = False
                    break
            if should_keep:
                selected_clusters.append(candidate)
                self._print_log(
                    f"  ✓ Candidate #{candidate['rank']} selected (Quality: {candidate['quality_score']:.3f})")
        return selected_clusters

    def predict(self, points_2d: np.ndarray, show=False, save_path=None):
        """
        Predict the cluster labels for the given points.

        Args:
            points (np.ndarray): Input points of shape (N, 2) where N is the number of points.

        Returns:
            np.ndarray: Cluster labels of shape (N,) where N is the number of points.
        """
        if self.debug:
            print(f"\n{'='*60}")
            print(f"GMM Clustering Point cloud size: {len(points_2d)}")
            print(f"{'='*60}")
        remaining_points = points_2d.copy()
        init_points_num = len(remaining_points)

        self._result_dic = self._init_out(points_2d)

        iteration = 0
        remaining_cluster_num = init_points_num
        while iteration < self.max_iterations and \
                remaining_cluster_num >= self.min_cluster_points:
            iteration += 1
            self._print_log(
                f"Iteration {iteration+1}: {len(remaining_points)} remaining points")
            self._print_log(f"Step 1: Creating density grid...")
            x_min = remaining_points[:, 0].min()
            y_min = remaining_points[:, 1].min()
            sorted_cells = self._create_density_grid(
                remaining_points, descending=True)
            if len(sorted_cells) == 0:
                print(f"Step 1: No grid cell with points, break")
                break

            self._print_log(
                f"Step 2: Finding top-{self.top_k_grids} densest grid cells...")

            top_cells = sorted_cells[:min(self.top_k_grids, len(sorted_cells))]

            self._print_log(
                f"Step 3: GMM clustering on each grid center...")

            candidate_clusters = []
            for rank, (cell, count) in enumerate(top_cells, 1):
                density_ratio = count / remaining_cluster_num
                self._print_log(
                    f"    #{rank}: Cell {cell}, density: {count}/{remaining_cluster_num} ({density_ratio*100:.1f}%)")
                grid_center = grid_idx_to_point(
                    cell, self.grid_size, (x_min, y_min))
                nearest_gmm, dis_to_gmm = self._select_gmm_for_grid_center(
                    grid_center)

                if nearest_gmm is None:
                    print(f"Step 3: No GMM found, skipping")
                    continue

                self._print_log(f"    Nearest GMM: "
                                f"Mean=[{nearest_gmm['mean_point'][0]:.3f}, {nearest_gmm['mean_point'][1]:.3f}], "
                                f"n_points={nearest_gmm['n_points']}, Distance={dis_to_gmm:.3f}m")
                self._result_dic["model_found"] = True
                try:
                    tmp = self._clustering_points(nearest_gmm,
                                                  remaining_points, init_points_num, dis_to_gmm, grid_center)
                    if tmp is None:
                        continue
                    self._result_dic["info"]["model_found"] = True
                    self._print_log(f"    Assigned: {tmp['n_assigned']} points, Ratio: {tmp['n_points_ratio']:.2f}, "
                                    f"Quality: {tmp['quality_score']:.3f}, Status: {'✓' if tmp['is_reasonable'] else '✗'}")
                    cluster_points = remaining_points[tmp['assigned_mask']].copy(
                    )
                    info = {
                        'grid_center': grid_center,
                        'gmm': nearest_gmm,
                        'distance_to_gmm': dis_to_gmm,
                        'points': cluster_points,
                        'rank': rank
                    }
                    info.update(tmp)
                    candidate_clusters.append(info)
                except Exception as e:
                    print(f"Step 3: GMM covariance is not positive definite, skipping")
                    continue

            selected_clusters = self._evaluate_cluster(candidate_clusters)
            if len(selected_clusters) == 0:
                print(f"  No clusters passed NMS, stopping")
                break
            best_candidate = selected_clusters[0]
            self._print_log(f"  Best cluster:")
            self._print_log(
                f"    Grid center: [{best_candidate['grid_center'][0]:.3f}, {best_candidate['grid_center'][1]:.3f}]")
            self._print_log(f"    Points: {best_candidate['n_assigned']}")
            self._print_log(
                f"    Quality: {best_candidate['quality_score']:.3f}")
            self._print_log(
                f"    Status: {'REASONABLE ✓' if best_candidate['is_reasonable'] else 'UNREASONABLE ✗'}")
            # Check if we should stop based on n_points validation

            best_points = best_candidate["gmm_n_points"]
            best_candidate_score = ModelScorer.compute_points_score(
                best_points, best_candidate['n_assigned'], ratio_low=0.1, ratio_high=5)
            if best_candidate_score == 0 and \
                    (not best_candidate['is_reasonable']):
                print(
                    f"  ⚠️  Too few points compared to GMM model, remaining points likely noise")
                print(f"     Stopping iteration")
                break
            # Store the best cluster
            tmp = self._out_dic(
                cluster=best_candidate['points'],
                gmm=best_candidate['gmm'],
                distance_to_gmm=best_candidate['distance_to_gmm'],
                cluster_center=best_candidate['grid_center'],
                quality_score=best_candidate['quality_score'],
                is_reasonable=best_candidate['is_reasonable'],
                mean_mahal=best_candidate['mean_mahal'], inlier_ratio=best_candidate['inlier_ratio'],
                points_score_to_gmm=best_candidate_score)

            self._result_dic["clusters"].append(tmp)

            self._print_log(
                f"   Stored Cluster {len(self._result_dic['clusters'])}:")
            self._print_log(f"    Points: {best_candidate['n_assigned']}")
            self._print_log(
                f"    Actual center: [{tmp['actual_center'][0]:.3f}, {tmp['actual_center'][1]:.3f}]")
            self._print_log(
                f"    Inlier ratio: {best_candidate['inlier_ratio']*100:.1f}%")
            self._print_log(
                f"    Mean Mahalanobis: {best_candidate['mean_mahal']:.3f}")
            remaining_points = remaining_points[~best_candidate['assigned_mask']]
            remaining_cluster_num = len(remaining_points)
        self._result_dic["info"]["remaining_points"] = remaining_points.tolist()
        if show or save_path is not None:
            self.vis(show=show, save_path=save_path)
        return self._result_dic["info"]["model_found"], len(self._result_dic["clusters"])

    def vis(self, show=False, save_path=None):
        if not self._result_dic["info"]["model_found"]:
            print("No model found, skip visualization")
            return
        if not show and save_path is None:
            print("No show or save path, skip visualization")
            return
        fig, ax = plt.subplots(figsize=(14, 10))
        clusters = self._result_dic["clusters"]
        colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
        for i, (cluster, color) in enumerate(zip(clusters, colors)):
            points = np.array(cluster["points"])
            points_num = cluster['points_num']
            status = '✓' if cluster['is_reasonable'] else '✗'
            score = cluster['quality_score']
            gmm_score = cluster['points_score_to_gmm']
            center = cluster['cluster_center']
            gmm = cluster['gmm']

            ax.scatter(points[:, 0], points[:, 1], c=[color], s=50, alpha=0.6,
                       label=f"Cluster {i+1} ({points_num} pts) {status} Score {score:.3f} GMM Score {gmm_score:.3f}", edgecolors='black', linewidth=0.5)
            # Plot grid center
            ax.scatter(center[0], center[1], c=[color], marker='X', s=300,
                       edgecolors='black', linewidth=2, zorder=10)

            # Plot GMM mean
            gmm_mean = gmm['mean_point']
            ax.scatter(gmm_mean[0], gmm_mean[1], c=[color], marker='s', s=200,
                       edgecolors='black', linewidth=2, alpha=0.7, zorder=9)

            # Draw covariance ellipses
            cov = gmm['covariance']
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(
                eigenvectors[1, 0], eigenvectors[0, 0]))
            for n_std in [1, 2, 3]:
                width, height = 2 * n_std * np.sqrt(eigenvalues)
                ellipse = Ellipse(center, width, height, angle=angle,
                                  facecolor='none', edgecolor=color,
                                  linewidth=2, linestyle='--', alpha=0.5)
                ax.add_patch(ellipse)
        # Plot remaining points
        remaining_points = np.array(
            self._result_dic["info"]["remaining_points"])
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
        all_points_num = self._result_dic["info"]["n_points"]
        stats_text = f"Total points: {all_points_num}\n"
        stats_text += f"Clusters: {len(clusters)}\n"
        stats_text += f"Assigned: {sum(c['points_num'] for c in clusters)}\n"
        stats_text += f"Unassigned: {len(remaining_points)}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                family='monospace')
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to: {save_path}")
        if show:
            plt.show()
        plt.close(fig)


def convert_model_to_grid_gmm(model):
    out = GridGMMModel()
    model_type = CovarianceType(model["covariance_type"])
    out.covariance_type = model_type
    out.mean_point = np.array(model["mean"])
    if model_type in [CovarianceType.TEMPLATE, CovarianceType.BIN_TEMPLATE]:
        cov = model["covariance"]
        out.template_model.covariance_avg = np.array(cov["avg"])
        out.template_model.covariance_median = np.array(cov["median"])
        point_info = model["n_points"]
        out.template_model.min_n_points = point_info["min"]
        out.template_model.max_n_points = point_info["max"]
        out.template_model.avg_n_points = point_info["mean"]
        out.template_model.median_n_points = point_info["median"]
        out.template_model.std_n_points = point_info["std"]
        out.template_model.point_use_preferred = point_info["point_use_preferred"]
        out.template_model.valid = True
    else:
        out.covariance = np.array(model["covariance"])
        out.n_points = model["n_points"]
    return out


def load_grid_templates_with_json(grid_models_json_path):
    """
    从 gmm_models_use 产生的 grid_models.json 中读取标准模板。

    返回:
        grid_templates: dict
            key: (row, col)
            value: list
        grid_size, grid_bounds: 方便你检查是否一致
    """
    with open(grid_models_json_path, "r") as f:
        data = json.load(f)
    grid_size = data["grid_size"]
    grid_bounds = data["grid_bounds"]
    model_grid_info = ModelGridInfo(
        grid_size, grid_bounds[0], grid_bounds[1], True)
    out_models = {}
    for key, models in data["models"].items():
        if key == "-2_6":
            a = 1
        row, col = map(int, key.split("_"))
        out_models[(row, col)] = []
        for model in models:
            template_model = convert_model_to_grid_gmm(model)
            out_models[(row, col)].append(template_model)
    return out_models, model_grid_info


if __name__ == "__main__":
    import os
    print("Loading GMM models...")
    models_dir = './data/VRU_Passing_B36_002_FK_0_0/train'
    models_dir = os.path.join(models_dir, "grid_models_new.json")
    grid_templates, model_grid_info = load_grid_templates_with_json(
        models_dir)

    debug = True
    predict_gmm = PredictGMM(model_grid_info, grid_templates, debug=debug)

    if debug:
        pcd_file = './data/VRU_Passing_B36_002_FK_0_0/test/1763112308900/46_1.pcd'
        base_name = os.path.basename(pcd_file).split(".")[0]
        output_file = f"./result/{base_name}.png"
        points_2d = read_pcd_and_extract_2d(pcd_file)
        if points_2d is None:
            print(f"Failed to extract 2D points from {pcd_file}")
            exit(0)
        predict_gmm.predict(points_2d, show=True)
        exit(0)

    data_dir = "./data/VRU_Passing_B36_002_FK_0_0/test"
    out_dir = os.path.join("./result/test_nn")
    os.makedirs(out_dir, exist_ok=True)
    for pcd_dir in os.listdir(data_dir):
        tmp_out_dir = os.path.join(out_dir, pcd_dir)
        os.makedirs(tmp_out_dir, exist_ok=True)
        for pcd_file_ in os.listdir(os.path.join(data_dir, pcd_dir)):
            if pcd_file_.endswith(".pcd"):
                base_name = os.path.basename(pcd_file_).split(".")[0]
                output_file = os.path.join(
                    tmp_out_dir, f"{base_name}.png")
                pcd_file = os.path.join(data_dir, pcd_dir, pcd_file_)
                print(f"Processing {pcd_file}...")
                points_2d = read_pcd_and_extract_2d(pcd_file)
                if points_2d is None:
                    continue
                predict_gmm.predict(points_2d, show=False,
                                    save_path=output_file)
