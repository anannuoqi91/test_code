import os
import numpy as np
import open3d as o3d
from cyber_record.record import Record
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import json
from box_obj import SIMPL_BOX_TYPE as BOX_TYPE


def write_pcd(points, out_path):
    points = np.array(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(out_path, pcd)


def get_all_files(in_record_dir, key_wors=""):
    all_files = [os.path.join(in_record_dir, f)
                 for f in os.listdir(in_record_dir) if key_wors in f]
    return all_files


def get_all_boxes(path, channel_name, obj_types=[]):
    new_type = [BOX_TYPE[t] for t in obj_types]
    record = Record(path)
    out = {}
    for topic, message, t in record.read_messages([channel_name]):
        idx = round(message.idx)
        tmp = []
        boxes = message.box
        for box in boxes:
            if (obj_types and box.object_type not in new_type) or len(box.point_index) == 0:
                continue
            tmp_box = {
                "track_id": int(box.track_id),
                "point_index": list(box.point_index),
            }
            tmp.append(tmp_box)
        if tmp:
            out[idx] = tmp
    return out


def multi_process_get_boxes(all_files, box_channel, use_type=["PEDESTRIAN"], max_workers=8):
    boxes = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(get_all_boxes, path, box_channel, use_type)
            for path in all_files
        ]

        for path, future in tqdm(zip(all_files, futures),
                                 total=len(all_files),
                                 desc="Processing boxes"):
            tmp = future.result()
            if tmp:
                boxes.update(tmp)
    return boxes


def parse_point_core_numpy(point_core_bytes):
    dtype = np.dtype([
        ('x', '<f4'),
        ('y', '<f4'),
        ('z', '<f4'),
        ('intensity', '<u2'),
        ('_pad', 'V2'),
        ('timestamp', '<u8'),
    ])
    return np.frombuffer(point_core_bytes, dtype=dtype)


def generate_pcd(path, boxes, channel_name, out_dir):
    record = Record(path)
    for topic, message, t in record.read_messages([channel_name]):
        idx = round(message.idx)
        if idx not in boxes:
            continue
        points_byte = message.point_core
        points = parse_point_core_numpy(points_byte)
        for box in boxes[idx]:
            track_id = box['track_id']
            os.makedirs(os.path.join(out_dir, f"{track_id}"), exist_ok=True)
            out_path = os.path.join(out_dir, f"{track_id}", f"{idx}.pcd")
            point_index = box["point_index"]
            box_points = points[point_index]
            xyz = np.column_stack(
                (box_points['x'], box_points['y'], box_points['z'])
            )
            write_pcd(xyz, out_path)


def write_boxes_to_json(boxes: dict, out_path: str):
    with open(out_path, 'w') as f:
        json.dump(boxes, f, indent=4)


def read_boxes_from_json(json_path: str):
    with open(json_path, 'r') as f:
        boxes = json.load(f)
    return boxes


if __name__ == '__main__':
    in_record_dir = '/home/seyond_user/od/SW/A12_FK_1'
    out_dir = './data/single_lidar/A12/'
    box_channels = ['omnisense/bkg/01/inlier_boxes']
    pcd_channel = 'omnisense/preprocess/01/parallel_up_dynamic_points'
    boxes_path = os.path.join(out_dir, 'boxes.json')
    all_files = get_all_files(in_record_dir, "a12")
    boxes = multi_process_get_boxes(all_files, box_channels[0], ["PEDESTRIAN"])
    write_boxes_to_json(boxes, boxes_path)
    # boxes = read_boxes_from_json(boxes_path)
    for f in all_files:
        generate_pcd(f, boxes, pcd_channel, out_dir)
