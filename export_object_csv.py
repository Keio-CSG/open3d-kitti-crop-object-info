import argparse
import os
from io import TextIOWrapper
import numpy as np
import open3d as o3d
from math import cos, sin
from object_info import ObjectInfo
import time

from kitti_utils import load_kitti_calib, read_objs2velo, class_list


def crop_kitti(data_path: str, out_file: str):
    file = open(out_file, "w")
    file.write(ObjectInfo.get_csv_header() + "\n")

    time_start = time.time()
    for frame in range(0, 7481):
        print(f"\rProcessing frame {frame+1} / 7481", end="")
        crop_kitti_frame(data_path, frame, file)
    time_end = time.time()
    print(f"\nTime elapsed: {time_end - time_start} seconds")

    file.close()


def crop_kitti_frame(data_path: str, frame: int, file: TextIOWrapper):

    # get data
    lidar_path = os.path.join(data_path, 'velodyne')
    label_path = os.path.join(data_path, 'label_2')
    calib_path = os.path.join(data_path, 'calib')

    lidar_file = os.path.join(lidar_path, f'{frame:06d}.bin')
    label_file = os.path.join(label_path, f'{frame:06d}.txt')
    calib_file = os.path.join(calib_path, f'{frame:06d}.txt')

    # get calibration
    calib = load_kitti_calib(calib_file)

    # get points and boxes
    boxes_velo, objs_type = read_objs2velo(label_file, calib['Tr_velo2cam'])
    points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    box = boxes_velo[0]
    for box, obj_type in zip(boxes_velo, objs_type):
        L = box[2]
        W = box[1]
        H = box[0]
        X = box[3]
        Y = box[4]
        Z = box[5]
        Theta = box[6]
        vol = o3d.visualization.SelectionPolygonVolume()
        vol.orthogonal_axis = "Z"
        vol.axis_max = Z + H
        vol.axis_min = Z
        vol.bounding_polygon = o3d.utility.Vector3dVector(np.array([
            [
                L/2*cos(Theta)-W/2*sin(Theta)+X,
                L/2*sin(Theta)+W/2*cos(Theta)+Y,
                0
            ],
            [
                L/2*cos(Theta)+W/2*sin(Theta)+X,
                L/2*sin(Theta)-W/2*cos(Theta)+Y,
                0
            ],
            [
                -L/2*cos(Theta)+W/2*sin(Theta)+X,
                -L/2*sin(Theta)-W/2*cos(Theta)+Y,
                0
            ],
            [
                -L/2*cos(Theta)-W/2*sin(Theta)+X,
                -L/2*sin(Theta)+W/2*cos(Theta)+Y,
                0
            ]
        ], dtype=np.float64))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd_crop = vol.crop_point_cloud(pcd)
        write_object_info(box, pcd_crop, obj_type, frame, file)


def write_object_info(
        box: list, pcd, obj_type: int, frame: int, file: TextIOWrapper):
    obj_type_name = class_list[obj_type]
    new_obj = ObjectInfo.create(box, pcd, obj_type_name, frame)
    file.write(new_obj.to_csv_line() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", "--d", type=str,
        default="./data/KITTI/training", help="kitti data path")
    # parser.add_argument(
    #     "--frame", "--f", type=int,
    #     default=500, help="frame of the data")
    parser.add_argument(
        "--out", "--o", type=str,
        default="./output.csv", help="output csv file")
    opt = parser.parse_args()

    data_path = opt.data
    # frame = opt.frame
    out_file = opt.out

    crop_kitti(data_path, out_file)
