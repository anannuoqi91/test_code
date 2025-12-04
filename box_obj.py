import os
try:
    import numpy as np
    import shapely.geometry as geo
except ImportError:
    pass


SIMPL_BOX_TYPE = {
    "UNKNOWN": 0,
    "PEDESTRIAN": 1,
    "CYCLIST": 2,
    "CAR": 3,
    "TRUCK": 4,
    "BUS": 5,
}


GT_BOX_TYPE = {
    "UNKNOWN": 0,
    "CAR": 1,
    "TRUCK": 2,
    "PEDESTRIAN": 3,
    "CYCLIST": 4,
    "CONE": 5,
    "OTHERS": 6,
    "BUS": 7,

}


def from_gt_txt(line_str, timestamp_ms=0):
    line_s = line_str.strip().split(' ')
    heading_radis = -float(line_s[6])
    tmp = {
        'track_id':  int(float(line_s[0])),
        'class_id': int(float(line_s[1])),
        'confidence': float(line_s[2]),
        'truncation': float(line_s[3]),
        'occlusion': float(line_s[4]),
        'rotation_y': float(line_s[5]),
        'heading_radians': heading_radis,
        'x': float(line_s[7]),
        'y': -float(line_s[8]),
        'z': float(line_s[9]),
        'l': float(line_s[10]),
        'w': float(line_s[11]),
        'h': float(line_s[12]),
        'timestamp_ms': timestamp_ms,
    }
    if len(line_s) > 16:
        tmp['class_confidence'] = float(line_s[13])
        tmp['speed_x'] = float(line_s[14])
        tmp['speed_y'] = float(line_s[15])
        tmp['speed_z'] = float(line_s[16])
    else:
        tmp['speed_x'] = float(line_s[13])
        tmp['speed_y'] = float(line_s[14])
        tmp['speed_z'] = float(line_s[15])
    return tmp


def box_conners(box, zkey="x"):
    cz = box[zkey]
    cy = box["y"]
    value_cos = np.cos(box["heading_radians"])
    value_sin = np.sin(box["heading_radians"])
    cos_y = value_cos * box["w"] / 2
    cos_z = value_cos * box["l"] / 2
    sin_y = value_sin * box["w"] / 2
    sin_z = value_sin * box["l"] / 2
    box_2d = [
        (cos_y + sin_z + cy, -sin_y + cos_z + cz),
        (cos_y - sin_z + cy, -sin_y - cos_z + cz),
        (-cos_y - sin_z + cy, sin_y - cos_z + cz),
        (-cos_y + sin_z + cy, sin_y + cos_z + cz)
    ]

    box_2d = [(i[0], i[1]) for i in box_2d]
    return box_2d


def box_conners_geo(box):
    return geo.Polygon(box_conners(box))


def get_box_points(pcd, box, zkey="x"):
    box_geo = box_conners_geo(box)
    box_pcd = []
    for p in pcd:
        if box_geo.contains(geo.Point(p[1], p[2])):
            box_pcd.append(p)
    return box_pcd


def read_txt_gt(file_path):
    out = []
    if not os.path.exists(file_path):
        return out
    with open(file_path, 'r') as file:
        line = file.readline()
        while line:
            tmp_box = from_gt_txt(line)
            out.append(tmp_box)
            line = file.readline()
    return out
