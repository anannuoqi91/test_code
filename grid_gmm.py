from typing import List
from global_info import *
import json


class GridGMM:
    def __init__(self, grid_info: ModelGridInfo,
                 models: List[ResultTrainer],
                 num_thresh: int = None,
                 cluster_points_thresh: int = 5):
        self.grid_size = grid_info.grid_size
        self.origin_y = grid_info.origin_y
        self.origin_z = grid_info.origin_z
        self.num_thresh = num_thresh if num_thresh is not None else max(
            int(2 * self.grid_size / 1.0), 5)
        self.cluster_points_thresh = cluster_points_thresh
        self._grid_gmm = {}
        self._models = []
        self.generate_grid_gmm(models)

    def _sort_by_conv(self, models: List[ResultTrainer]):
        all_covs = [np.asarray(params.covariance, dtype=float)
                    for params in models]
        # 协方差的迹，等价于两个方向方差之和，代表“整体大小”，比 det 稳定一些
        sizes = [float(np.trace(c)) for c in all_covs]
        idx_sorted = np.argsort(sizes)  # 从小到大排序的下标
        return idx_sorted

    def _sort_by_score(self, models: List[ResultTrainer]):
        """
        计算网格GMM模型的分数
        1. 先根据模型的 self_score 排序
        2. 再根据模型的 axis_ratio 排序
        3. 最后根据模型的 points_num 排序
        4. 1～3的分数相乘，作为最终的分数
        """
        all_scores = [params.self_score for params in models]
        axis_ratio = [ModelScorer.axis_ratio(
            np.asarray(params.covariance, dtype=float)) for params in models]
        axis_ratio_stat = choose_center_stat(axis_ratio)
        axis_ratio_score = [cal_value_score(
            ratio, axis_ratio_stat) for ratio in axis_ratio]
        points_num = [params.n_points for params in models]
        center_stat = choose_center_stat(points_num)
        base_num = center_stat[center_stat["preferred"]]
        points_num_score = [ModelScorer.compute_points_score(
            base_num, points_num) for points_num in points_num]
        final_scores = [a * b * c for a, b, c in zip(
            all_scores, axis_ratio_score, points_num_score)]
        idx_sorted = np.argsort(final_scores)  # 从小到大排序的下标
        return idx_sorted

    def _rm_grid_outlier_by_score(self, models: List[ResultTrainer]):
        idx_sorted = self._sort_by_score(models)
        idx_sorted = idx_sorted[1:-1]
        models = [models[i] for i in idx_sorted]
        return models

    def _grid_cov_stats(self, cov_list):
        """
        cov_list: list[np.ndarray], 每个是 2x2 协方差矩阵

        返回:
            {
                "sigma_min": {mean, std, cv},
                "sigma_max": {mean, std, cv},
                "axis_ratio": {mean, std, cv},
                "angle": {
                    "mean": float,   # 平均主轴方向 (弧度)
                    "cv": float,     # 用圆方差 1 - R 来表示离散度
                },
            }
        """
        cov_arr = np.stack(cov_list, axis=0)  # (M, 2, 2)
        M = cov_arr.shape[0]

        sigma_min = np.empty(M, dtype=float)
        sigma_max = np.empty(M, dtype=float)
        angles = np.empty(M, dtype=float)

        for i in range(M):
            cov = np.asarray(cov_arr[i], dtype=float)

            # 特征值 + 特征向量
            vals, vecs = np.linalg.eigh(cov)
            vals = np.maximum(vals, 1e-12)  # 防止负数/零

            order = np.argsort(vals)       # [λ_min, λ_max]
            vals = vals[order]
            vecs = vecs[:, order]

            sigma_min[i] = np.sqrt(vals[0])
            sigma_max[i] = np.sqrt(vals[1])

            # 主轴方向 (长轴，对应 λ_max，那就是 vecs[:,1])
            angle = np.arctan2(vecs[1, 1], vecs[0, 1])
            angles[i] = angle

        ratio = sigma_max / (sigma_min + 1e-12)  # 长短轴比

        def _summary(x):
            return {
                "mean": float(x.mean()),
                "std": float(x.std()),
                "cv": float(x.std() / (x.mean() + 1e-12)),
            }

        # 圆统计: angles 弧度
        sin_sum = np.sin(angles).mean()
        cos_sum = np.cos(angles).mean()
        R = np.sqrt(sin_sum**2 + cos_sum**2)  # 0~1, 越接近1越集中

        mean_angle = np.arctan2(sin_sum, cos_sum)  # 平均方向 (弧度)
        circ_var = 1.0 - R                         # 当作“cv”用

        return {
            "sigma_min": _summary(sigma_min),
            "sigma_max": _summary(sigma_max),
            "axis_ratio": _summary(ratio),
            "angle": {
                "mean": float(mean_angle),
                "cv": float(circ_var),
            },
        }

    def _is_stable_grid(self,
                        stats, model_num,
                        # 大小稳定性：主/副轴 σ 的 cv 上限
                        sigma_cv_thresh=0.3,
                        # 形状稳定性：长宽比的 std 上限
                        ratio_std_thresh=0.5,
                        # 方向稳定性：angle.cv(圆方差) 上限
                        angle_cv_thresh=0.5,
                        num_thresh=10,
                        ):
        """
        根据 grid_cov_stats 的统计结果，判断一个网格是否“稳定单峰”，
        如果稳定则适合用一个标准模板 (μ_grid, Σ_grid)。

        参数
        ----
        stats : dict
            来自 grid_cov_stats(cov_list) 的结果：
            {
                "sigma_min": {"mean", "std", "cv"},
                "sigma_max": {"mean", "std", "cv"},
                "axis_ratio": {"mean", "std", "cv"},
                "angle": {"mean", "cv"},
            }
        sigma_cv_thresh : float
            主/副轴 σ 的变异系数阈值（越小越严格）。
        ratio_std_thresh : float
            长宽比的标准差阈值（越小越严格）。
        angle_cv_thresh : float
            方向的圆方差阈值，0~1，越小表示方向越集中。

        返回
        ----
        is_stable : bool
            True: 认为该网格“稳定单峰”，可用单一标准模板；
            False: 认为该网格波动大/多模态，不建议只用一个模板。
        """

        sigma_max_cv = stats["sigma_max"]["cv"]
        sigma_min_cv = stats["sigma_min"]["cv"]
        ratio_std = stats["axis_ratio"]["std"]
        angle_cv = stats["angle"]["cv"]

        cond_size = (sigma_max_cv < sigma_cv_thresh) and (
            sigma_min_cv < sigma_cv_thresh)
        cond_shape = (ratio_std < ratio_std_thresh)
        cond_angle = (angle_cv < angle_cv_thresh)
        cond_num = (model_num > num_thresh)

        return cond_size and cond_shape and cond_angle and cond_num

    def _aggregate_grid_stats(self, covs,
                              use_median=False,
                              shrinkage=0.1,
                              min_eig=1e-6):
        """
        给定一个网格里的所有 mean 和 cov，计算该网格的“标准模板” (μ_grid, Σ_grid)。

        point2d: 中心点  对应 YZ 平面上的均值
        covs : list[np.ndarray]  shape (N, 2, 2)

        use_median: True 则用中位数，否则用平均
        shrinkage: 协方差收缩系数 λ，0 表示不收缩，0.1 表示 Σ' = (1-λ)*Σ + λ*αI

        返回:
            mu_grid:  (2,) ndarray
            cov_grid: (2,2) ndarray (已做对称化 + 特征值下限 + shrink) 转list
        """
        covs_arr = np.stack(covs, axis=0)         # (N, 2, 2)
        # 2) cov 聚合：简单按元素平均或中位数
        if use_median:
            # 把 (N,2,2) 展开成 (N,4)，每个元素取中位数，再 reshape 回 2x2
            cov_flat = covs_arr.reshape(covs_arr.shape[0], -1)
            cov_med_flat = np.median(cov_flat, axis=0)
            cov_grid = cov_med_flat.reshape(2, 2)
        else:
            cov_grid = np.mean(covs_arr, axis=0)

        # 3) 保证协方差对称
        cov_grid = 0.5 * (cov_grid + cov_grid.T)

        # 4) 特征值分解，做最小特征值裁剪 + shrink
        vals, vecs = np.linalg.eigh(cov_grid)
        vals = np.maximum(vals, min_eig)  # 防止出现非正定

        if shrinkage > 0.0:
            # shrink 到 αI 上，α 用当前特征值平均
            alpha = float(vals.mean())
            vals = (1.0 - shrinkage) * vals + shrinkage * alpha

        cov_grid = vecs @ np.diag(vals) @ vecs.T

        return cov_grid.tolist()

    def _build_grid_standard_templates(self, models_list,
                                       shrinkage=0.1,
                                       ):
        """
        use_median : bool
            True: mean/cov 用中位数聚合；False: 用平均数聚合。
        shrinkage : float
            协方差收缩系数 λ，0 ~ 1。0 表示不收缩，0.1 是一个比较保守的值。

        返回:
        ----
        grid_templates : dict
            key: (row, col)
            value: {
                "mean": np.ndarray shape (2,),      # μ_grid
                "covariance": {},
                "count": int,                      # 该网格参与聚合的模型数量
            }
        """
        grid_templates = TemplateModel()
        cov_list = [m.covariance for m in models_list]
        grid_templates.covariance_avg = self._aggregate_grid_stats(
            cov_list,
            use_median=False,
            shrinkage=shrinkage,
        )
        grid_templates.covariance_median = self._aggregate_grid_stats(
            cov_list,
            use_median=True,
            shrinkage=shrinkage,
        )
        n_points_list = [m.n_points for m in models_list]
        out = choose_center_stat(n_points_list)
        grid_templates.avg_n_points = int(out["mean"])
        grid_templates.median_n_points = int(out["median"])
        grid_templates.min_n_points = int(out["min"])
        grid_templates.max_n_points = int(out["max"])
        grid_templates.std_n_points = out["std"]
        grid_templates.point_use_preferred = out["preferred"]

        return grid_templates

    def _create_template_model(self, type_: CovarianceType,
                               mean_point: List[float],
                               **kwargs):
        models_list = kwargs.get("models_list", None)
        covariance = kwargs.get("covariance", None)
        n_points = kwargs.get("n_points", None)
        grid_gmm = GridGMMModel()
        grid_gmm.covariance_type = type_
        if models_list is not None:
            grid_gmm.template_model = self._build_grid_standard_templates(
                models_list)
        if covariance is not None and n_points is not None:
            grid_gmm.n_points = n_points
            grid_gmm.covariance = covariance
        grid_gmm.mean_point = mean_point
        return grid_gmm

    def _split_bin_by_trace(self, models: List[ResultTrainer],
                            num_bins=4):
        """
        将模型按 trace 到中位数的距离，分为 num_bins 个 bin
        """
        all_covs = [np.asarray(c.covariance, dtype=float) for c in models]
        traces = np.array([float(np.trace(c)) for c in all_covs])
        median_trace = np.median(traces)
        deviations = np.abs(traces - median_trace)
        idx_sorted = np.argsort(deviations)
        num = len(idx_sorted)
        base = num // num_bins
        remainder = num % num_bins
        start = 0
        tmp_bin_models = []
        for bin_id in range(num_bins):
            size = base + (1 if bin_id < remainder else 0)
            if size == 0:
                continue
            end = start + size
            bin_idx = idx_sorted[start:end]
            start = end
            cov_bin = [all_covs[k] for k in bin_idx]
            models_bin = [models[k] for k in bin_idx]
            points_bin = [models[k].mean_point for k in bin_idx]
            stats_bin = self._grid_cov_stats(cov_bin)
            stable_bin = self._is_stable_grid(
                stats_bin, len(cov_bin), num_thresh=0
            )
            if stable_bin:
                model_bin = self._create_template_model(
                    CovarianceType.BIN_TEMPLATE,
                    np.mean(np.array(points_bin), axis=0).tolist(),
                    models_list=models_bin
                )
                tmp_bin_models.append(model_bin)
        return tmp_bin_models

    def generate_grid_gmm(self, models: List[ResultTrainer]):
        """
        生成网格GMM模型
        """
        if len(models) == 0:
            print("No models to calculate grid idx.")
            return
        bins_dict = {}
        for params in models:
            if not params.update or not params.converged:
                continue
            row, col = grid_to_bin_idx(
                np.array(params.mean_point), self.grid_size, (self.origin_y, self.origin_z))
            bins_dict.setdefault((row, col), []).append(params)
        for key in bins_dict.keys():
            if key == (-2, 6):
                a = 1
            if key == (-2, 7):
                a = 1
            models_list = bins_dict[key]
            if len(models_list) == 0:
                continue
            if len(models_list) > self.num_thresh:
                idx_sorted = self._sort_by_score(models_list)
                idx_sorted = idx_sorted[1:-1]
                models_list = [models_list[i] for i in idx_sorted]
            if len(models_list) > self.num_thresh:
                idx_sorted = self._sort_by_conv(models_list)
                idx_sorted = idx_sorted[1:-1]
                models_list = [models_list[i] for i in idx_sorted]
            cov_list = [np.asarray(params.covariance, dtype=float)
                        for params in models_list]
            stats = self._grid_cov_stats(cov_list)
            mean_yz = grid_idx_to_point(
                key, self.grid_size, (self.origin_y, self.origin_z))
            stable = self._is_stable_grid(stats,
                                          len(cov_list),
                                          num_thresh=self.num_thresh)
            if stable:
                self._grid_gmm[key] = [self._create_template_model(
                    CovarianceType.TEMPLATE, mean_yz.tolist(), models_list=models_list)]
                continue
            # 这里使用最终的 final_cov_list / final_n_points_list 作为基础
            if len(models_list) < self.num_thresh:
                tmp_models = []
                for model_ in models_list:
                    model_type = CovarianceType(model_.covariance_type)
                    tmp_models.append(self._create_template_model(
                        model_type, model_.mean_point, n_points=model_.n_points, covariance=model_.covariance))
                self._grid_gmm[key] = tmp_models
                continue
            # 3-2) 若数量 >= num_thresh:
            #  以 trace 为尺度, 看每个模型到 trace 中位数的偏差, 按偏差划分为4个bin
            tmp_bin_models = self._split_bin_by_trace(models_list, num_bins=4)
            if len(tmp_bin_models) == 0:
                continue
            self._grid_gmm[key] = tmp_bin_models

    def save_grid_gmm(self, out_file):
        """
        保存网格GMM模型
        """
        out = {}
        out["grid_size"] = self.grid_size
        out["grid_bounds"] = [self.origin_y, self.origin_z]
        out["models"] = {}
        for key, v in self._grid_gmm.items():
            out["models"][f"{key[0]}_{key[1]}"] = []
            for i_m in v:
                tmp = {
                    "mean": i_m.mean_point,
                    "covariance_type": i_m.covariance_type.value,
                }
                if i_m.template_model.valid:
                    i_m_model = i_m.template_model
                    tmp["covariance"] = {
                        "avg": i_m_model.covariance_avg,
                        "median": i_m_model.covariance_median
                    }
                    tmp["n_points"] = {
                        "mean": i_m_model.avg_n_points,
                        "median": i_m_model.median_n_points,
                        "min": i_m_model.min_n_points,
                        "max": i_m_model.max_n_points,
                        "std": i_m_model.std_n_points,
                        "point_use_preferred": i_m_model.point_use_preferred
                    }
                else:
                    tmp["covariance"] = i_m.covariance
                    tmp["n_points"] = i_m.n_points
                out["models"][f"{key[0]}_{key[1]}"].append(tmp)

        with open(out_file, 'w') as f:
            json.dump(out, f, indent=4)


def load_all_gmm_models(model_files, cal_type=None):
    """
    Load all GMM models from JSON files in the data directory

    Returns:
        all_models: List of dictionaries with GMM parameters
    """
    print(f"Found {len(model_files)} JSON files:")
    all_models = []

    for json_file in sorted(model_files):
        try:
            with open(json_file, 'r') as f:
                models = json.load(f)

            for filename, params in models.items():
                model_info = ResultTrainer(**params)
                model_info.info = json_file
                if cal_type is not None:
                    if cal_type == "all":
                        cal_type = ["distance", "angle", "grid"]
                    else:
                        cal_type = [cal_type]
                    if "distance" in cal_type:
                        model_info["distance"] = compute_distance_from_origin_2d(
                            model_info["mean"])
                        model_info["distance_bin"] = assign_distance_bin(
                            model_info["distance"], DISTANCE_BINS)
                    if "angle" in cal_type:
                        model_info["angle"] = compute_angle_from_origin_2d(
                            model_info["mean"])
                        model_info["angle_bin"] = assign_angle_bin(
                            model_info["angle"], N_ANGLE_SECTORS)
                    if "grid" in cal_type:
                        model_info["grid_idx"] = grid_to_bin_idx(
                            model_info["mean"], GRID_SIZE, GRID_BOUNDS)
                all_models.append(model_info)

            print(f"  Loaded {len(models)} models from {json_file}")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    print(f"\nTotal models loaded: {len(all_models)}")
    return all_models


if __name__ == "__main__":
    import glob
    import os
    models_dir = './data/VRU_Passing_B36_002_FK_0_0/train'
    json_pattern = os.path.join(models_dir, "**/gmm_models.json")
    json_files = glob.glob(json_pattern, recursive=True)
    if len(json_files) == 0:
        print(f"No gmm_models.json files found in ./data")
        exit(1)
    gmm_models = load_all_gmm_models(json_files)
    grid_gmm = GridGMM(ModelGridInfo(5.0, 0, 0), gmm_models)
    grid_gmm.save_grid_gmm(os.path.join(models_dir, "grid_models_new_.json"))
