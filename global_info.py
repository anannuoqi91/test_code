import numpy as np
import math


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
