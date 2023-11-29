import numpy as np


class ObjectInfo:
    @classmethod
    def get_csv_header(cls) -> str:
        return (
            "type,frame,distance_m,"
            "min_azimuth_deg,max_azimuth_deg,"
            "min_altitude_deg,max_altitude_deg,"
            "min_distance_m,max_distance_m,"
            "height,width,length,"
            "center_x,center_y,center_z,"
            "yaw_rad,points_count"
        )

    @classmethod
    def create(cls, box: list, pcd, obj_type: str, frame: int) -> "ObjectInfo":
        L = box[2]
        W = box[1]
        H = box[0]
        X = box[3]
        Y = box[4]
        Z = box[5]
        Theta = box[6]

        points_numpy = np.asarray(pcd.points)  # (n, 3)
        if points_numpy.shape[0] > 0:
            # xyz to spherical coordinates
            r = np.linalg.norm(points_numpy, axis=1)
            theta = np.arctan2(points_numpy[:, 1], points_numpy[:, 0])
            phi = np.arccos(points_numpy[:, 2] / r)

            min_azimuth_deg = np.rad2deg(np.min(theta))
            max_azimuth_deg = np.rad2deg(np.max(theta))
            min_altitude_deg = np.rad2deg(np.min(phi))
            max_altitude_deg = np.rad2deg(np.max(phi))
            min_distance_m = np.min(r)
            max_distance_m = np.max(r)
            point_num = points_numpy.shape[0]
        else:
            min_azimuth_deg = 0
            max_azimuth_deg = 0
            min_altitude_deg = 0
            max_altitude_deg = 0
            min_distance_m = 0
            max_distance_m = 0
            point_num = 0

        return cls(
            type=obj_type,
            frame=frame,
            distance_m=float(np.linalg.norm([X, Y, Z])),
            min_azimuth_deg=min_azimuth_deg,
            max_azimuth_deg=max_azimuth_deg,
            min_altitude_deg=min_altitude_deg,
            max_altitude_deg=max_altitude_deg,
            min_distance_m=min_distance_m,
            max_distance_m=max_distance_m,
            height=H,
            width=W,
            length=L,
            center_x=X,
            center_y=Y,
            center_z=Z,
            yaw_rad=Theta,
            points_count=point_num,
        )

    def __init__(
            self,
            type: str,
            frame: int,
            distance_m: float,
            min_azimuth_deg: float,
            max_azimuth_deg: float,
            min_altitude_deg: float,
            max_altitude_deg: float,
            min_distance_m: float,
            max_distance_m: float,
            height: float,
            width: float,
            length: float,
            center_x: float,
            center_y: float,
            center_z: float,
            yaw_rad: float,
            points_count: int,):
        self.type = type
        self.frame = frame
        self.distance_m = distance_m
        self.min_azimuth_deg = min_azimuth_deg
        self.max_azimuth_deg = max_azimuth_deg
        self.min_altitude_deg = min_altitude_deg
        self.max_altitude_deg = max_altitude_deg
        self.min_distance_m = min_distance_m
        self.max_distance_m = max_distance_m
        self.height = height
        self.width = width
        self.length = length
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.yaw_rad = yaw_rad
        self.points_count = points_count

    def to_csv_line(self) -> str:
        return (
            f"{self.type},{self.frame},{self.distance_m:.3f},"
            f"{self.min_azimuth_deg:.3f},{self.max_azimuth_deg:.3f},"
            f"{self.min_altitude_deg:.3f},{self.max_altitude_deg:.3f},"
            f"{self.min_distance_m:.3f},{self.max_distance_m:.3f},"
            f"{self.height:.3f},{self.width:.3f},{self.length:.3f},"
            f"{self.center_x:.3f},{self.center_y:.3f},{self.center_z:.3f},"
            f"{self.yaw_rad:.3f},{self.points_count}"
        )
