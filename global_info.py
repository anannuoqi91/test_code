from dis import distb
import numpy as np
import math
import open3d as o3d
from typing import List
from enum import Enum


class CovarianceType(Enum):
    DIAG = "diag"
    FULL = "full"
    TEMPLATE = "template"
    BIN_TEMPLATE = "bin_template"
    SPHERICAL = "spherical"
    UNKNOWN = "unknown"


class ResultTrainer:
    mean_point: List[float] = []
    covariance: List[List[float]] = [[]]
    covariance_type: str = ""
    n_points: int = 0
    converged: bool = False
    n_iter: int = 0
    log_likelihood: float = 0.0
    aic: float = 0.0
    bic: float = 0.0
    update: bool = False
    max_iter: int = 0
    self_score: float = 0.0
    info: str = ""

    def to_dict(self):
        return self.__dict__

    def __setattr__(self, name, value):
        """
        重写属性设置方法，当修改任意属性时自动将update设置为True
        """
        # 调用父类的__setattr__方法设置属性
        super().__setattr__(name, value)

        # 如果修改的不是update属性本身，则自动设置update为True
        if name != 'update':
            super().__setattr__('update', True)

    def __init__(self, **kwargs):
        """
        初始化方法，确保在初始化时也能正确设置update状态
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.update = len(kwargs) > 0


class TemplateModel:
    def __setattr__(self, name, value):
        """
        重写属性设置方法，当修改任意属性时自动将valid设置为True
        """
        # 调用父类的__setattr__方法设置属性
        super().__setattr__(name, value)

        # 如果修改的不是valid属性本身，则自动设置valid为True
        if name != 'valid':
            super().__setattr__('valid', True)

    def __init__(self, **kwargs):
        """
        初始化方法，确保在初始化时也能正确设置valid状态
        """
        self.covariance_avg = np.zeros((2, 2), dtype=float)
        self.covariance_median = np.zeros((2, 2), dtype=float)
        self.min_n_points = 0
        self.max_n_points = 0
        self.avg_n_points = 0.0
        self.median_n_points = 0.0
        self.std_n_points = 0.0
        self.point_use_preferred = None

        for key, value in kwargs.items():
            setattr(self, key, value)
        self.valid = len(kwargs) > 0

    def to_dict(self):
        return self.__dict__


class ModelGridInfo:
    grid_size: float = 0.0
    origin_y: float = 0.0
    origin_z: float = 0.0
    valid: bool = False

    def __setattr__(self, name, value):
        """
        重写属性设置方法，当修改任意属性时自动将valid设置为True
        """
        # 调用父类的__setattr__方法设置属性
        super().__setattr__(name, value)

        # 如果修改的不是valid属性本身，则自动设置valid为True
        if name != 'valid':
            super().__setattr__('valid', True)

    def __init__(self, grid_size=0.0, origin_y=0.0, origin_z=0.0, valid=False):
        """
        初始化方法，确保在初始化时也能正确设置valid状态
        """
        self.grid_size = grid_size
        self.origin_y = origin_y
        self.origin_z = origin_z
        self.valid = valid


class GridGMMModel:
    def __init__(self):
        self.covariance = np.zeros((2, 2), dtype=float)
        self.mean_point = np.zeros(2, dtype=float)
        self.covariance_type = CovarianceType.UNKNOWN
        self.template_model = TemplateModel()
        self.n_points = 0


class ModelScorer:
    def __init__(self, min_points_abs=5,
                 ratio_low=0.2,
                 ratio_high=2.0,
                 ratio_good_max=8.0,
                 ratio_bad_max=8.0):
        self.min_points_abs = min_points_abs
        self.ratio_low = ratio_low
        self.ratio_high = ratio_high
        self.ratio_good_max = ratio_good_max
        self.ratio_bad_max = ratio_bad_max

    @staticmethod
    def mahalanobis_dist2(points, mean_point, cov):
        """
        points: (N,2)
        mean: (2,)
        cov: (2,2)
        返回: 每个点的 d^2
        """
        mean_point = np.array(mean_point)
        points = np.array(points)
        diff = points - mean_point.reshape(1, 2)
        cov_inv = np.linalg.inv(cov)
        # (N,2) @ (2,2) -> (N,2), 再逐行点乘
        tmp = diff @ cov_inv
        d2 = np.sum(tmp * diff, axis=1)
        return d2

    @staticmethod
    def inlier_score_by_sigma(points_2d,
                              params: ResultTrainer,
                              k_sigma=3.0,
                              good_outlier_frac=0.02,  # <=2% 视为很好
                              bad_outlier_frac=0.2):  # >=20% 视为很差
        """
        基于 3σ 椭圆外的点比例打一个 [0,1] 分数：
        - outlier_frac <= good_outlier_frac -> 1.0
        - outlier_frac >= bad_outlier_frac  -> 0.0
        - 中间线性插值 [0.0, 1.0]
        """
        if len(points_2d) < 3:
            return 0.0

        d2 = ModelScorer.mahalanobis_dist2(
            points_2d, params.mean_point, params.covariance)
        thr = (k_sigma ** 2)  # 2D 情况下, 3σ 椭圆对应 d^2=9
        outlier_frac = float(np.mean(d2 > thr))

        if outlier_frac <= good_outlier_frac:
            return 1.0
        if outlier_frac >= bad_outlier_frac:
            return 0.0

        # 线性从 1 降到 0
        score = (bad_outlier_frac - outlier_frac) / \
            (bad_outlier_frac - good_outlier_frac)
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def axis_ratio(cov):
        """
        计算椭圆的长宽比：sigma_max / sigma_min
        """
        eigvals, _ = np.linalg.eigh(cov)
        lam_min = float(np.minimum(eigvals[0], eigvals[1]))
        lam_max = float(np.maximum(eigvals[0], eigvals[1]))
        if lam_min <= 0.0:
            return 0.0
        sigma_min = np.sqrt(lam_min)
        sigma_max = np.sqrt(lam_max)
        return sigma_max / (sigma_min + 1e-12)

    def _shape_score_axis_ratio(self, cov):
        """
        根据椭圆长宽比打 [0,1] 分数：
        - axis_ratio <= ratio_good_max  -> 1.0
        - axis_ratio >= ratio_bad_max   -> 0.0
        - 中间线性下降
        """
        if self.ratio_bad_max == self.ratio_good_max:
            return 1.0
        axis_ratio = ModelScorer.axis_ratio(cov)
        if axis_ratio <= 0.0:
            return axis_ratio

        if axis_ratio <= self.ratio_good_max:
            return 1.0
        if axis_ratio >= self.ratio_bad_max:
            return 0.0

        score = (self.ratio_bad_max - axis_ratio) / \
            (self.ratio_bad_max - self.ratio_good_max)
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def compute_points_score(base_points_num, n_pts, min_points_abs=5, ratio_low=0.2, ratio_high=2.0):
        if base_points_num <= 0:
            return 1.0
        if n_pts < min_points_abs:
            return 0.0
        ratio = n_pts / float(base_points_num)
        if ratio <= ratio_low:
            return 0.0
        if ratio >= ratio_high:
            return 0.0
        # 中间部分：分两段线性上升 + 下降
        if ratio <= 1.0:
            # 从 ratio_low 到 1.0 之间线性上升到 1
            return (ratio - ratio_low) / (1.0 - ratio_low)
        else:
            # 从 1.0 到 ratio_high 之间线性下降到 0
            return (ratio_high - ratio) / (ratio_high - 1.0)

    def _score_single_model(self, params: ResultTrainer):
        """
        根据单个 GMM 拟合结果打一个 0~1 的质量分。
        分数主要用于“是否大致可靠”的判断，真正的模型选择
        （比如不同 K 比较）仍建议用相对评分（多模型一起归一化）。

        params: fit_gmm_single_file 返回的 dict（至少包含 converged, n_iter, log_likelihood, aic, bic）
        max_iter: 当时 GMM 设定的 max_iter
        """
        # 2) 迭代次数得分：越少越好
        iter_ratio = params.n_iter / float(params.max_iter)
        iter_ratio = min(max(iter_ratio, 0.0), 1.0)   # 截断到 [0,1]
        iter_score = 1.0 - iter_ratio                # 少迭代 → 分高
        # 3) log_likelihood / AIC / BIC 这里**不做绝对归一化**
        #    单模型没有可比基准，只做一些“极端情况”的惩罚
        ll = params.log_likelihood
        aic = params.aic
        bic = params.bic

        ll_score = 1.0
        if np.isfinite(ll):
            ll_clipped = np.clip(ll, -1e4, 0.0)
            ll_score = (ll_clipped + 1e4) / 1e4

        aic_score = 1.0 if np.isfinite(aic) else 0.0
        bic_score = 1.0 if np.isfinite(bic) else 0.0

        w_iter = 0.8
        w_ll = 0.1
        w_ic = 0.1

        core_score = (
            w_iter * iter_score +
            w_ll * ll_score +
            w_ic * 0.5 * (aic_score + bic_score)
        )
        core_score = float(np.clip(core_score, 0.0, 1.0))
        return core_score

    def compute_score(self, params: ResultTrainer,
                      base_points_num: int = 0,
                      points_2d=None):
        default_score = 0.0
        if not params.update or not params.converged:
            return default_score
        if params.self_score > 0.0:
            self_score = params.self_score
        else:
            core_score = self._score_single_model(params)
            disturb_score = 1
            if points_2d is not None:
                disturb_score = ModelScorer.inlier_score_by_sigma(
                    points_2d, params)
            self_score = core_score * disturb_score
        if base_points_num == 0:
            base_points_num = params.n_points
        points_num_score = ModelScorer.compute_points_score(
            base_points_num, params.n_points, self.min_points_abs, self.ratio_low, self.ratio_high)
        shape_score = self._shape_score_axis_ratio(params.covariance)
        final_score = self_score * points_num_score * shape_score
        final_score = float(np.clip(final_score, 0.0, 1.0))
        return final_score


def choose_center_stat(values, delta_thresh=0.2, n_small=20):
    values = np.asarray(values)
    mean = values.mean()
    median = np.median(values)
    std = values.std(ddof=0)

    eps = 1e-12
    delta = abs(mean - median) / (std + eps)

    n = len(values)

    # 默认用 mean
    preferred = "mean"

    # 条件 1：偏斜比较明显
    if delta > delta_thresh:
        preferred = "median"

    # 条件 2：样本太少，优先中位数
    if n < n_small:
        preferred = "median"

    return {
        "mean": mean,
        "median": median,
        "std": std,
        "delta": delta,
        "preferred": preferred,
        "min": np.min(values),
        "max": np.max(values),
    }


def cal_value_score(value, center_stat, thres=10, eps=1e-12):
    """
    计算一个值相对于中心统计量的“好程度”：
      - 刚好等于中心 -> 1.0 (最好)
      - 偏离达到 thres * std -> 0.0
      - 中间线性下降
      - 超过阈值继续视为 0

    center_stat: 之前 choose_center_stat 返回的 dict
        - center_stat["preferred"]: "mean" 或 "median"
        - center_stat["mean"], center_stat["median"], center_stat["std"]
    """
    center = center_stat[center_stat["preferred"]]
    std = center_stat["std"]

    # std 太小时要特殊处理，避免除零
    if std < eps:
        # 所有值都几乎一样：
        #   如果 value 也≈center，就给满分；否则直接 0
        return 1.0 if abs(value - center) <= thres * eps else 0.0

    dist = abs(value - center)           # 距离中心有多远
    norm = dist / (thres * std)          # 归一化到 "多少个 thres 内"

    if norm >= 1.0:
        # 超过 thres * std，视为最差
        return 0.0

    # [0, 1) 上线性映射：距离 0 -> score=1，距离 thres*std -> score=0
    score = 1.0 - norm
    return float(max(0.0, min(1.0, score)))


# ---------------- 距离分 bin ----------------
DISTANCE_BINS = [i for i in range(0, 71, 10)] + [1e9]
GRID_SIZE = 10.0
GRID_BOUNDS = [0.0, 0.0]


def compute_distance_from_origin_2d(point_2d: np.ndarray) -> float:
    """
    根据 2D 坐标计算距离，这里使用 mean 的 [Y,Z] 欧氏距离即可。
    """
    return float(np.linalg.norm(point_2d))


def assign_distance_bin(distance: float, bins: list[float]) -> int:
    """
    根据 DISTANCE_BINS 将距离映射到某个 bin 的索引。
    约定区间为: [bins[i], bins[i+1]) 映射到 bin i。
    """
    import bisect

    # 找到第一个 > distance 的位置
    idx = bisect.bisect_right(bins, distance) - 1
    # clamp 到合法范围 [0, len(bins)-2]
    idx = max(0, min(idx, len(bins) - 2))
    return int(idx)


# ---------------- 角度分 bin 配置 ----------------
N_ANGLE_SECTORS = 8


def compute_angle_from_origin_2d(point_2d: np.ndarray) -> float:
    """
    根据 2D 坐标计算角度：
    这里 point_2d = [Y, Z]，角度定义为 atan2(Z, Y)，范围 [-pi, pi)。
    如果你以后换成 [X,Y] 平面，只要改这里即可。
    """
    y, z = float(point_2d[0]), float(point_2d[1])
    return float(math.atan2(z, y))


def assign_angle_bin(angle_deg: float, n_sectors: int) -> int:
    """
    将角度(单位: 度)映射到 [0, n_sectors) 的扇区编号。

    约定:
    - 角度可以是任意值, 会自动归一化到 [-180, 180) 等价的 [0, 360) 进行计算
    - 扇区 0 对应区间: [-w/2, w/2), 其中 w = 360 / n_sectors
    - 扇区 1: [w/2, 3w/2), 依此类推, 在圆上环形分布
    """
    width = 360.0 / n_sectors          # 每个扇区的宽度 w
    a0 = angle_deg % 360.0             # 先归一化到 [0, 360)
    # 左右各半扇区中心对齐: 让 [-w/2, w/2) 对应扇区 0
    a_shift = (a0 + width / 2.0) % 360.0
    sector = int(a_shift // width)     # 0..n_sectors-1
    return sector


def sector_bounds_deg(sector_idx: int, n_sectors: int):
    """
    给定扇区编号和扇区总数, 返回该扇区在 [-180, 180) 上的角度区间 [start_deg, end_deg)。

    注意:
    - 对于跨越 -180/180 的扇区(比如 8 区时的 sector 4),
      返回值会是类似 (157.5, -157.5),
      表示区间 [157.5, 180) U [-180, -157.5)
    """
    width = 360.0 / n_sectors
    center = sector_idx * width                # 以 0° 为扇区 0 中心
    start = center - width / 2.0
    end = center + width / 2.0

    def norm180(a):
        return ((a + 180.0) % 360.0) - 180.0   # 归一化到 [-180, 180)

    start_n = norm180(start)
    end_n = norm180(end)
    return start_n, end_n


def grid_to_bin_idx(point_2d: np.ndarray, grid_size, bounds: tuple[float, float]):
    y, z = float(point_2d[0]), float(point_2d[1])
    min_y, min_z = bounds
    row = int((y - min_y) // grid_size)
    col = int((z - min_z) // grid_size)
    return row, col


def grid_idx_to_point(grid_idx, grid_size, bounds) -> np.ndarray:
    row, col = grid_idx
    min_y, min_z = bounds
    y = min_y + (row + 0.5) * grid_size
    z = min_z + (col + 0.5) * grid_size
    return np.array([y, z])


def read_pcd_and_extract_2d(filename,
                            filter_di_m=1000.0,
                            filter_points_num=5):
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
        valid_mask = (np.abs(points_2d) < filter_di_m).all(axis=1)
        points_2d = points_2d[valid_mask]

        if len(points_2d) < filter_points_num:
            return None

        return points_2d

    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None


def write_pcd(points, out_path):
    points = np.array(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(out_path, pcd)


if __name__ == "__main__":
    print("角度分区范围（度）：")
    for i in range(N_ANGLE_SECTORS):
        print(f"扇区{i}: {sector_bounds_deg(i, N_ANGLE_SECTORS)}")

    print("\n测试角度映射：")
    test_angles_deg = [-180, -157.5, -112.5, -
                       67.5, -22.5, 0, 22.5, 67.5, 112.5, 157.5, 180]
    for angle_deg in test_angles_deg:
        bin_idx = assign_angle_bin(angle_deg, N_ANGLE_SECTORS)

        print(f"角度 {angle_deg:6.1f}° -> 扇区 {bin_idx}")
