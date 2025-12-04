import json
import os
import numpy as np
import copy
from global_info import *

model_dic_template = {
    "covariance": [],  # or {}
    "n_points": 0,  # or {}
    "mean": [],
    "covariance_type": "diag",
}


def _aggregate_grid_stats(covs, use_median=False, shrinkage=0.1, min_eig=1e-6):
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


def build_grid_standard_templates(cov_list,
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
    grid_templates = copy.deepcopy(model_dic_template)
    grid_templates["covariance"] = {
        "avg": _aggregate_grid_stats(
            cov_list,
            use_median=False,
            shrinkage=shrinkage,
        ),
        "median": _aggregate_grid_stats(
            cov_list,
            use_median=True,
            shrinkage=shrinkage,
        ),
    }
    grid_templates["covariance_type"] = "template"
    return grid_templates


def grid_cov_stats(cov_list):
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


def circular_stats(angles):
    """
    angles: ndarray, 弧度
    返回: 平均角度 和 "圆方差"
    """
    sin_sum = np.sin(angles).mean()
    cos_sum = np.cos(angles).mean()
    R = np.sqrt(sin_sum**2 + cos_sum**2)  # 0~1, 越接近1越集中

    mean_angle = np.arctan2(sin_sum, cos_sum)
    # 一个常用的圆方差定义
    circ_var = 1 - R

    return float(mean_angle), float(circ_var)


def covariance_prototype_and_spread(cov_list):
    cov_arr = np.stack(cov_list, axis=0)  # (M, 2, 2)
    # 简单用平均协方差作为 prototype
    cov_proto = cov_arr.mean(axis=0)

    diffs = cov_arr - cov_proto
    # Frobenius norm
    dists = np.linalg.norm(diffs.reshape(len(cov_arr), -1), axis=1)

    return cov_proto, {
        "mean": float(dists.mean()),
        "std": float(dists.std()),
        "max": float(dists.max()),
    }


def kl_gaussian(mu0, cov0, mu1, cov1):
    """
    KL( N0 || N1 ), 2D 高斯
    """
    d = mu0.shape[0]
    cov1_inv = np.linalg.inv(cov1)
    diff = mu1 - mu0

    term1 = np.trace(cov1_inv @ cov0)
    term2 = diff.T @ cov1_inv @ diff
    term3 = np.log(np.linalg.det(cov1) / np.linalg.det(cov0) + 1e-12)
    return 0.5 * (term1 + term2 - d + term3)


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
                model_info = {
                    'mean': np.array(params['mean']),
                    'covariance': np.array(params['covariance']),
                    'covariance_type': params['covariance_type'],
                    'n_points': params['n_points'],
                    'filename': filename,
                    'json_source': json_file,
                }
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


def is_stable_grid(
    stats, model_num,
    # 大小稳定性：主/副轴 σ 的 cv 上限
    sigma_cv_thresh=0.3,
    # 形状稳定性：长宽比的 std 上限
    ratio_std_thresh=0.3,
    # 方向稳定性：angle.cv(圆方差) 上限
    angle_cv_thresh=0.3,
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


def rm_min_max_conv(cov_list, points_num_list, points_list):
    all_covs = [np.asarray(cov, dtype=float) for cov in cov_list]
    # 协方差的迹，等价于两个方向方差之和，代表“整体大小”，比 det 稳定一些
    sizes = [float(np.trace(c)) for c in all_covs]
    idx_sorted = np.argsort(sizes)  # 从小到大排序的下标
    keep_idx = idx_sorted[1:-1]             # 去掉最小的和最大的各一个
    cov_list = [all_covs[k] for k in keep_idx]
    points_num_list = [points_num_list[k] for k in keep_idx]
    points_list = [points_list[k] for k in keep_idx]
    return cov_list, points_num_list, points_list


def rm_min_points(cov_list, points_num_list, points_list, thresh=5):
    keep_idx = [i for i, n in enumerate(points_num_list) if n >= thresh]
    cov_list = [cov_list[i] for i in keep_idx]
    points_num_list = [points_num_list[i] for i in keep_idx]
    points_list = [points_list[i] for i in keep_idx]
    return cov_list, points_num_list, points_list


def split_bin_by_trace(cov_list, points_num_list, points_list, num_bins=4):
    """
    将模型按 trace 到中位数的距离，分为 num_bins 个 bin
    """
    all_covs = [np.asarray(c, dtype=float) for c in cov_list]
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
        n_points_bin = [points_num_list[k] for k in bin_idx]
        points_bin = [points_list[k] for k in bin_idx]
        stats_bin = grid_cov_stats(cov_bin)
        stable_bin = is_stable_grid(
            stats_bin, len(cov_bin), num_thresh=0
        )
        if stable_bin:
            model_bin = build_grid_standard_templates(cov_bin)
            model_bin["mean"] = np.mean(np.array(points_bin), axis=0).tolist()
            model_bin["n_points"] = {
                "min": float(min(n_points_bin)),
                "max": float(max(n_points_bin)),
                "mean": float(np.mean(n_points_bin)),
                "median": float(np.median(n_points_bin)),
                "std": float(np.std(n_points_bin)),
            }
            model_bin["covariance_type"] = "bin_template"
            tmp_bin_models.append(model_bin)
    return tmp_bin_models


def grid_models(model_files,
                grid_size,
                grid_bounds=(0.0, 0.0),
                num_thresh=None, cluster_points_thresh=5):
    """
    将 GMM models 分配到网格中
    """
    if num_thresh is None:
        num_thresh = min(int(grid_size / 1.0), 5)
    if len(model_files) == 0:
        print(f"No gmm_models.json files found in ./data")
        return
    gmm_models = load_all_gmm_models(model_files)
    if isinstance(gmm_models, dict):
        models_list = list(gmm_models.values())
    else:
        models_list = list(gmm_models)
    if len(models_list) == 0:
        print("No models to calculate grid idx.")
        return
    bins_dict = {}
    max_row = 0
    max_col = 0
    min_row = 0
    min_col = 0
    out = {}
    out["grid_size"] = grid_size
    out["grid_bounds"] = grid_bounds
    out["models"] = {}
    for params in models_list:
        row, col = grid_to_bin_idx(
            np.array(params["mean"]), grid_size, grid_bounds)
        cov = np.array(params["covariance"], dtype=float)
        bins_dict.setdefault((row, col), []).append(
            [cov, params["n_points"], params["mean"]])
        max_row = max(max_row, row)
        max_col = max(max_col, col)
        min_row = min(min_row, row)
        min_col = min(min_col, col)
    for i in range(min_row, max_row + 1):
        for j in range(min_col, max_col + 1):
            key = (i, j)
            model_list = bins_dict.get(key, [])
            if len(model_list) == 0:
                continue
            cov_list, n_points_list, points_list = zip(*model_list)
            cov_list = list(cov_list)
            n_points_list = list(n_points_list)
            points_list = list(points_list)
            if len(cov_list) > num_thresh:
                cov_list, n_points_list, points_list = rm_min_max_conv(
                    cov_list, n_points_list, points_list)
            stats = grid_cov_stats(cov_list)
            mean_yz = grid_idx_to_point((i, j), grid_size, grid_bounds)

            stable = is_stable_grid(stats, len(
                cov_list), num_thresh=num_thresh)
            if not stable:
                pre_num = len(cov_list)
                cov_list, n_points_list, points_list = rm_min_points(
                    cov_list, n_points_list, points_list, thresh=cluster_points_thresh)
                tmp_num = len(cov_list)
                if tmp_num >= 2 and tmp_num < pre_num:
                    stats2 = grid_cov_stats(cov_list)
                    stable = is_stable_grid(stats2, len(
                        cov_list), num_thresh=num_thresh)

            if stable:
                model = build_grid_standard_templates(cov_list)
                model["mean"] = [mean_yz[0], mean_yz[1]]
                model["n_points"] = {"min": min(n_points_list),
                                     "max": max(n_points_list),
                                     "mean": np.mean(n_points_list),
                                     "median": np.median(n_points_list),
                                     "std": np.std(n_points_list),
                                     }
                out["models"][f"{i}_{j}"] = [model]
                continue
            # =============== 新增策略三: stable 仍为 False =================
            # 这里使用最终的 final_cov_list / final_n_points_list 作为基础
            num_models = len(cov_list)
            if num_models < num_thresh:
                tmp_models = []
                for j in range(num_models):
                    tmp = copy.deepcopy(model_dic_template)
                    tmp["covariance"] = cov_list[j].tolist()
                    tmp["n_points"] = n_points_list[j]
                    tmp["mean"] = points_list[j].tolist()
                    tmp_models.append(tmp)
                out["models"][f"{i}_{j}"] = tmp_models
                continue
            # 3-2) 若数量 >= num_thresh:
            #  以 trace 为尺度, 看每个模型到 trace 中位数的偏差, 按偏差划分为4个bin
            tmp_bin_models = split_bin_by_trace(
                cov_list, n_points_list, points_list, num_bins=4)
            if len(tmp_bin_models) == 0:
                continue
            out["models"][f"{i}_{j}"] = tmp_bin_models
    return out


def load_grid_templates(grid_models_json_path):
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

    grid_size = data.get("grid_size", GRID_SIZE)
    grid_bounds = data.get("grid_bounds", GRID_BOUNDS)

    templates = {}

    for key, model in data["models"].items():
        row, col = map(int, key.split("_"))
        templates[(row, col)] = model

    print(f"[load_grid_templates] loaded {len(templates)} grid templates")
    return templates, grid_size, grid_bounds


if __name__ == "__main__":
    import glob
    import os
    models_dir = './data_/single_lidar'
    json_pattern = os.path.join(models_dir, "**/gmm_models.json")
    json_files = glob.glob(json_pattern, recursive=True)

    if len(json_files) == 0:
        print(f"No gmm_models.json files found in ./data")
    else:
        out = grid_models(json_files, grid_size=3, grid_bounds=(0, 0))
        with open(os.path.join(models_dir, "grid_models.json"), 'w') as f:
            json.dump(out, f, indent=4)
