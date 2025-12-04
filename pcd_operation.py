import open3d as o3d
import numpy as np


def write_pcd(points, out_path):
    points = np.array(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(out_path, pcd)


def convert_gt_to_simpl(in_file, out_file):
    """
    Read PCD file and extract 2D coordinates [Y, Z]

    Returns:
        points_2d: 2D point cloud [Y, Z] or None if failed
    """
    try:
        pcd = o3d.io.read_point_cloud(in_file)
        points_3d = np.asarray(pcd.points)

        if points_3d.shape[0] == 0 or points_3d.shape[1] < 3:
            return None

        # Extract [Y, X] coordinates (map to X, Y)
        points_3d = np.column_stack(
            [points_3d[:, 2], -1 * points_3d[:, 1], points_3d[:, 0]])

        write_pcd(points_3d, out_file)
        return points_3d

    except Exception as e:
        print(f"Error reading {in_file}: {e}")
        return None


if __name__ == "__main__":
    import os
    data_dir = '/home/demo/下载/VRU_Passing_B36_002_FK_0_0'
    pcd_dir = os.path.join(data_dir, "pcd")
    simpl_pcd_dir = os.path.join(data_dir, "simpl_pcd")
    os.makedirs(simpl_pcd_dir, exist_ok=True)
    for i in os.listdir(pcd_dir):
        if not i.endswith(".pcd"):
            continue
        convert_gt_to_simpl(os.path.join(pcd_dir, i),
                            os.path.join(simpl_pcd_dir, i))
