import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from gmm_models_use import *
from global_info import *


def visualize_gmm_covariance_by_grid(
    gmm_models,
    fig_size_per_cell=8.0,
    out_dir=None
):
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    # 1) 统一 gmm_models 结构为 list[dict]
    if isinstance(gmm_models, dict):
        models_list = list(gmm_models.values())
    else:
        models_list = list(gmm_models)

    if len(models_list) == 0:
        print("[visualize_gmm_covariance_by_bins] No models to visualize.")
        return

    bins_dict = {}
    max_row = 0
    max_col = 0
    min_row = 0
    min_col = 0
    for params in models_list:
        row, col = grid_to_bin_idx(
            np.array(params["mean"]), GRID_SIZE, GRID_BOUNDS)
        meanyz = grid_idx_to_point((row, col), GRID_SIZE, GRID_BOUNDS)
        max_row = max(max_row, row)
        max_col = max(max_col, col)
        min_row = min(min_row, row)
        min_col = min(min_col, col)
        cov = np.array(params["covariance"], dtype=float)
        bins_dict.setdefault((row, col), []).append((cov, meanyz))

    # 8) 绘制每个模型的协方差椭圆
    #    - 每个 (db, ab) 最多画 max_samples_per_bin 个
    #    - 以 mean 的取整为中心
    for i in range(min_row, max_row + 1):
        for j in range(min_col, max_col + 1):
            key = (i, j)
            cov_list = bins_dict.get(key, [])
            if len(cov_list) < 5:
                continue
            all_covs = [np.asarray(cov, dtype=float) for cov, _ in cov_list]
            all_means = [meanyz for _, meanyz in cov_list]
            # 协方差的迹，等价于两个方向方差之和，代表“整体大小”，比 det 稳定一些
            sizes = [float(np.trace(c)) for c in all_covs]
            idx_sorted = np.argsort(sizes)          # 从小到大排序的下标
            keep_idx = idx_sorted[1:-1]             # 去掉最小的和最大的各一个
            cov_list_plot = [all_covs[k] for k in keep_idx]
            meanyz_list = [all_means[k] for k in keep_idx][0]

            fig, ax = plt.subplots(
                figsize=(fig_size_per_cell, fig_size_per_cell))
            ax.set_aspect("equal")
            ax.set_xlabel("Y")
            ax.set_ylabel("Z")
            ax.set_title(f"GMM covariance by grid ({i}, {j})")
            ax.grid(True, linestyle=":", alpha=0.3)
            angle_list = []
            for cov in cov_list_plot:
                cov = np.asarray(cov, dtype=float)
                # 以 mean 的取整为中心
                center = (0, 0)

                # 协方差特征值分解，得到长短轴和旋转角
                vals, vecs = np.linalg.eigh(cov)
                vals = np.maximum(vals, 1e-12)  # 防止负数/零
                order = np.argsort(vals)
                vals = vals[order]
                vecs = vecs[:, order]
                angle = np.arctan2(vecs[1, 1], vecs[0, 1])
                angle_list.append(angle)

                # 画 2σ 椭圆：width = 4*sqrt(λ_max), height = 4*sqrt(λ_min)
                width = 4.0 * np.sqrt(vals[1])
                height = 4.0 * np.sqrt(vals[0])
                angle_deg = np.degrees(angle)

                e = Ellipse(
                    xy=(center[0], center[1]),
                    width=width,
                    height=height,
                    angle=angle_deg,
                    fill=False,
                    linewidth=0.5,
                    alpha=0.5,
                )
                ax.add_patch(e)
            all_cov = np.stack(cov_list_plot, axis=0)
            vals_all = np.linalg.eigvalsh(all_cov)
            all_cov = np.stack(cov_list_plot, axis=0)
            vals_all = np.linalg.eigvalsh(all_cov)
            limit = 3.0 * np.sqrt(vals_all.max())
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            stats = grid_cov_stats(cov_list_plot)
            ax.text(
                0.01,
                0.99,
                "models: {}\n"
                "mean: {}\n"
                "σ_max: μ={:.3f}, cv={:.2f}\n"
                "σ_min: μ={:.3f}, cv={:.2f}\n"
                "ratio: μ={:.2f}, σ={:.2f}\n"
                "angle: μ={:.2f}, cv={:.2f}\n".format(
                    len(cov_list_plot),
                    meanyz_list,
                    stats["sigma_max"]["mean"], stats["sigma_max"]["cv"],
                    stats["sigma_min"]["mean"], stats["sigma_min"]["cv"],
                    stats["axis_ratio"]["mean"], stats["axis_ratio"]["std"],
                    np.degrees(circular_stats(angle_list)[0]),
                    circular_stats(angle_list)[1],
                ),
                fontsize=8,
                transform=ax.transAxes,
                va="top",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="white", ec="gray", alpha=0.8),
            )

            plt.tight_layout()
            if out_dir is not None:
                plt.savefig(os.path.join(
                    out_dir, f"grid_{i}_{j}_modelnum_{len(cov_list_plot)}.png"))
            else:
                plt.show()


if __name__ == "__main__":
    import glob
    import os
    json_pattern = os.path.join('./data', "**/gmm_models.json")
    json_files = glob.glob(json_pattern, recursive=True)

    if len(json_files) == 0:
        print(f"No gmm_models.json files found in ./data")
    else:
        gmm_models = load_all_gmm_models(json_files)
        visualize_gmm_covariance_by_grid(
            gmm_models, fig_size_per_cell=8, out_dir="./data/vis_gmm"
        )
