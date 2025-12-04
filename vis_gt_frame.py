from box_obj import *
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np


def read_gt_pcd_and_extract_2d(filename):
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

        # Extract [Y, X] coordinates (map to X, Y)
        points_2d = np.column_stack([-1*points_3d[:, 1], points_3d[:, 0]])

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


def vis_gt_frame(pcd_path, box_path, box_type=["PEDESTRIAN"]):
    box_type = [GT_BOX_TYPE[bt] for bt in box_type]
    boxes = read_txt_gt(box_path)
    fig, ax = plt.subplots(figsize=(16, 16))
    for box in boxes:
        if box["class_id"] not in box_type:
            continue
        color_str = 'red'
        pts = box_conners(box)
        x = []
        y = []
        for pt in pts:
            x.append(pt[0])
            y.append(pt[1])
        x.append(pts[0][0])
        y.append(pts[0][1])
        center_x = np.mean(y)
        center_y = np.mean(x)
        ax.plot(y, x, color=color_str)
        ax.text(center_x, center_y, str(box["track_id"]),
                fontsize=8, ha='center', va='center')
    pcd_points = read_gt_pcd_and_extract_2d(pcd_path)
    new_points = pcd_points[(pcd_points[:, 1] < 50)]
    ax.scatter(new_points[:, 0], new_points[:, 1], s=1, color='gray')
    plt.show()


if __name__ == "__main__":
    data_dir = '/home/demo/下载/VRU_Passing_B36_002_FK_0_0'
    pcd_path = os.path.join(data_dir, "pcd", "1763112308400.pcd")
    box_path = os.path.join(data_dir, "box", "1763112308400.txt")
    vis_gt_frame(pcd_path, box_path, box_type=["PEDESTRIAN"])
