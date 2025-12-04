import os
import copy
from collections import defaultdict
from global_info import *

from box_obj import GT_BOX_TYPE as BOX_TYPE
from box_obj import from_gt_txt, get_box_points
from pcd_operation import convert_gt_to_simpl


def read_gt_pcd_to_simpl(filename):
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
        points_3d = np.column_stack(
            [points_3d[:, 2], -1 * points_3d[:, 1], points_3d[:, 0]])

        return points_3d

    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None


def _distance(box1, box2):
    return np.linalg.norm(np.array([box1['y'], box1['z']]) - np.array([box2['y'], box2['z']]))


def read_box(file_path):
    out = []
    with open(file_path, "r") as f:
        line = f.readline()
        while line:
            box_tmp = from_gt_txt(line)
            if box_tmp['class_id'] == BOX_TYPE['PEDESTRIAN']:
                out.append(copy.deepcopy(box_tmp))
            line = f.readline()
    return out


def split_boxes(boxes, threshold=0.3):
    out = defaultdict(list)
    num = len(boxes)
    used = [False] * num
    for i in range(num):
        for j in range(i+1, num):
            if used[j]:
                continue
            if _distance(boxes[i], boxes[j]) < threshold:
                out[boxes[i]['track_id']].append(boxes[j])
                used[j] = True
        out[boxes[i]['track_id']].append(boxes[i])
        used[i] = True
    return out


def generate_pcd(pcd_path, box_path, out_dir):
    boxes = read_box(box_path)
    if len(boxes) == 0:
        return
    if len(boxes) > 1:
        out = split_boxes(boxes)
    else:
        out = {boxes[0]['track_id']: boxes}
    pcd = o3d.io.read_point_cloud(pcd_path)
    points_3d = np.asarray(pcd.points)
    for key, v in out.items():
        box_pcd = []
        for box in v:
            box_pcd.extend(get_box_points(points_3d, box))
        if len(box_pcd) == 0:
            continue
        out_path = os.path.join(out_dir, f"{key}_{len(v)}.pcd")
        write_pcd(np.asarray(box_pcd), out_path)


def generate_pcd_one(pcd_path, box_path, out_dir):
    boxes = read_box(box_path)
    if len(boxes) == 0:
        return
    pcd = o3d.io.read_point_cloud(pcd_path)
    points_3d = np.asarray(pcd.points)
    for box in boxes:
        track_id = box["track_id"]
        box_pcd = get_box_points(points_3d, box)
        if len(box_pcd) == 0:
            continue
        out_path = os.path.join(out_dir, f"{track_id}.pcd")
        write_pcd(np.asarray(box_pcd), out_path)


if __name__ == "__main__":
    # Configuration
    data_dir = '/home/demo/下载/VRU_Passing_B36_002_FK_0_0'
    pcd_dir = os.path.join(data_dir, "pcd")
    box_dir = os.path.join(data_dir, "box")
    simpl_pcd_dir = os.path.join(data_dir, "simpl_pcd")
    # os.makedirs(simpl_pcd_dir, exist_ok=True)
    # for i in os.listdir(pcd_dir):
    #     if not i.endswith(".pcd"):
    #         continue
    #     convert_gt_to_simpl(os.path.join(pcd_dir, i),
    #                         os.path.join(simpl_pcd_dir, i))
    base_name = os.path.basename(data_dir)
    out_dir = f"./data/{base_name}"
    os.makedirs(out_dir, exist_ok=True)
    num = 10
    all_pcd_file = [i for i in os.listdir(simpl_pcd_dir) if i.endswith(".pcd")]
    # 随机取 80%
    np.random.shuffle(all_pcd_file)
    train_pcd_file = all_pcd_file[:int(len(all_pcd_file) * 0.8)]
    # for i in train_pcd_file:
    #     tmp_dir = os.path.join(out_dir, "train", i.replace(".pcd", ""))
    #     os.makedirs(tmp_dir, exist_ok=True)
    #     generate_pcd_one(
    #         pcd_path=os.path.join(simpl_pcd_dir, i),
    #         box_path=os.path.join(box_dir, i.replace(".pcd", ".txt")),
    #         out_dir=tmp_dir
    #     )
    test_pcd_file = all_pcd_file[int(len(all_pcd_file) * 0.8):]
    for i in test_pcd_file:
        tmp_dir = os.path.join(out_dir, "test", i.replace(".pcd", ""))
        os.makedirs(tmp_dir, exist_ok=True)
        generate_pcd(
            pcd_path=os.path.join(simpl_pcd_dir, i),
            box_path=os.path.join(box_dir, i.replace(".pcd", ".txt")),
            out_dir=tmp_dir
        )
    # pcd = read_gt_pcd_to_simpl(os.path.join(
    #     data_dir, "pcd", "1763112308400.pcd"))
    # new_pcd = o3d.geometry.PointCloud()
    # new_pcd.points = o3d.utility.Vector3dVector(pcd)
    # o3d.io.write_point_cloud(os.path.join(
    #     out_dir, "1763112308400.pcd"), new_pcd)
