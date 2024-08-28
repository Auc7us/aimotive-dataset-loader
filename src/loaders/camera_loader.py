import json
import os
from typing import Dict, Tuple, List

import cv2
import numpy as np

from src.camera_params import CameraParams
from src.transform import Transform
import copy
import quaternion

class CameraData:
    """ Stores the CameraDataItem for each camera.

    Attributes
        front_wide_camera: image and camera parameters for the front wide camera.
        front_narrow_camera: image and camera parameters for the front narrow camera.
        left_camera: image and camera parameters for the left camera.
        right_camera: image and camera parameters for the right camera.
    """

    def __init__(self, front_wide_img: np.array, front_narrow_img: np.array, left_img: np.array,
                 right_img: np.array, camera_params_by_sensor: Dict[str, CameraParams]):
        """
        Args:
            front_wide_img: front wide image as numpy array
            front_narrow_img: front narrow image as numpy array
            left_img: left image as numpy array
            right_img: right image as numpy array
            camera_params_by_sensor: dict, key: sensor name, value: CameraParams
        """
        self.front_wide_camera = CameraDataItem('front_wide_cam', front_wide_img, camera_params_by_sensor['F_MIDRANGECAM_C'])
        self.front_narrow_camera = CameraDataItem('front_narrow_cam', front_narrow_img, camera_params_by_sensor['F_LONGRANGECAM_C'])
        self.left_camera = CameraDataItem('left_cam', left_img, camera_params_by_sensor['F_CTCAM_L'])
        self.right_camera = CameraDataItem('right_cam', right_img, camera_params_by_sensor['F_CTCAM_R'])

        self.index = 0
        self.items = [self.front_wide_camera, self.front_narrow_camera, self.left_camera, self.right_camera]

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == len(self.items):
            self.index = 0
            raise StopIteration

        item = self.items[self.index]
        self.index += 1

        return item


class CameraDataItem:
    """
    Stores the image and camera parameters for a camera with a given frame id.

    Attributes
        name: camera name
        image: images stored as a numpy array
        camera_params: camera parameters for the given data item
    """

    def __init__(self, name: str, image: np.array, camera_params: CameraParams):
        """
        Args:
            name: camera name
            image: images stored as a numpy array
            camera_params: camera parameters for the given data item
        """
        self.name = name
        self.image = image
        self.camera_params = camera_params

class CameraParameters:
    def __init__(self, **kwargs):
        for k, v, in kwargs.items():
            setattr(self, k, v)

def load_camera_data(data_folder: str, frame_id: str) -> CameraData:
    """
    Loads data for each camera with a given frame id.

    Args:
        data_folder: the path of the sequence from where data is loaded
        frame_id: id if the loadable camera data, e.g. 0033532

    Returns:
        camera_data: a CameraData instance with front wide, front narrow, left, right image and camera parameters.
    """

    front_wide_cam_path, front_narrow_cam_path, left_cam_path, right_cam_path = get_camera_paths(data_folder, frame_id)
    cali_path = os.path.join(data_folder, 'sensor', 'calibration', "calibration.json")
    camera_params_by_sensor = read_camera_params(cali_path)

    front_wide_img, front_narrow_img = cv2.imread(front_wide_cam_path), cv2.imread(front_narrow_cam_path)
    left_img, right_img = cv2.imread(left_cam_path), cv2.imread(right_cam_path)

    camera_data = CameraData(front_wide_img, front_narrow_img, left_img, right_img, camera_params_by_sensor)

    return camera_data

def read_camera_params(calipath):
    with open(calipath, 'r') as cali_in_file:
        sensors = json.load(cali_in_file)

    for sensor in sensors.values():
        sensor["device_id"] = str(sensor["device_id"])

    cameras = {sensor["label"]: sensor for sensor in sensors.values() if sensor["sensor_type"] == "camera"}
    for cam in cameras.values():
        calc_camparams(cam)
    return cameras

def calc_camparams(cam):
    imgproc_camparams = copy.deepcopy(cam)
    imgproc_camparams.update({'pos_meter': [0.0, 0.0, 0.0],
                              'yaw_pitch_roll_deg': [0.0, 0.0, 0.0],
                              'extrinsic_error': -1})
    imgproc_camparams = CameraParameters(**imgproc_camparams)
    cam["imgproc_camparams"] = imgproc_camparams

    yaw, pitch, roll = np.radians(cam["yaw_pitch_roll_deg"])
    R_cam_to_body = euler_to_matrix(-roll, -pitch, -yaw, order="XYZ")

    cam_ori_quat = quaternion.from_rotation_matrix(R_cam_to_body.T)
    camera_trafo = Transform(cam_ori_quat, cam["pos_meter"])
    cam['camera_transform'] = camera_trafo

    RT_cam_to_body = np.copy(R_cam_to_body)
    RT_cam_to_body[3, :3] = cam["pos_meter"]
    A_cam_to_view = np.array([[0, 0, 1, 0],
                              [-1, 0, 0, 0],
                              [0, -1, 0, 0],
                              [0, 0, 0, 1]], dtype=np.float64)
    body_to_view = Transform.RT_inverse(RT_cam_to_body) @ A_cam_to_view
    cam["body_to_view"] = body_to_view


def get_camera_paths(data_folder: str, frame_id: str) -> Tuple[str, str, str, str]:
    """
    Collects the path of the image with a given frame id for each camera.

    Args:
        data_folder: the path of the sequence from where data is loaded
        frame_id: id if the loadable camera data, e.g. 0033532

    Returns:
        a tuple with the path of the front, back, left, and right camera, respectively
    """
    cam_base_path = os.path.join(data_folder, 'sensor', 'camera')
    front_wide_cam_path = os.path.join(cam_base_path, 'F_MIDRANGECAM_C', 'F_MIDRANGECAM_C_' + frame_id + '.jpg')
    front_narrow_cam_path = os.path.join(cam_base_path, 'F_LONGRANGECAM_C', 'F_LONGRANGECAM_C_' + frame_id + '.jpg')
    left_cam_path = os.path.join(cam_base_path, 'F_CTCAM_L', 'F_CTCAM_L_' + frame_id + '.jpg')
    right_cam_path = os.path.join(cam_base_path, 'F_CTCAM_R', 'F_CTCAM_R_' + frame_id + '.jpg')

    return front_wide_cam_path, front_narrow_cam_path, left_cam_path, right_cam_path

def euler_to_matrix(x_rotation, y_rotation, z_rotation, order="XYZ", vectormode="row"):
    # example model coordinate system: x - forward (roll), y - left (pitch), z - up (yaw)
    # points are multiplied from the right: points_transformed = points @ R
    # order is: first multiply with roll, then pitch, then yaw
    assert vectormode in ["row", "column"]
    Ax = np.array([[1, 0, 0, 0],
                   [0, np.cos(x_rotation), np.sin(x_rotation), 0],
                   [0, -np.sin(x_rotation), np.cos(x_rotation), 0],
                   [0, 0, 0, 1]])

    Ay = np.array([[np.cos(y_rotation), 0, -np.sin(y_rotation), 0],
                   [0, 1, 0, 0],
                   [np.sin(y_rotation), 0, np.cos(y_rotation), 0],
                   [0, 0, 0, 1]])

    Az = np.array([[np.cos(z_rotation), np.sin(z_rotation), 0, 0],
                   [-np.sin(z_rotation), np.cos(z_rotation), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    if order == "XYZ":
        R = Ax @ Ay @ Az
    elif order == "XZY":
        R = Ax @ Az @ Ay
    elif order == "YXZ":
        R = Ay @ Ax @ Az
    elif order == "YZX":
        R = Ay @ Az @ Ax
    elif order == "ZXY":
        R = Az @ Ax @ Ay
    elif order == "ZYX":
        R = Az @ Ay @ Ax
    else:
        raise ValueError("Order \"{}\" is not supported!".format(order))
    if vectormode == "row":
        return R
    else:
        return R.T



