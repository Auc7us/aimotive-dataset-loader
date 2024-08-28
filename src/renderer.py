from typing import Tuple, List, Dict

import cv2
import numpy as np
from src.annotation import Annotation
from src.camera_params import CameraParams
from src.data_loader import DataItem
from src.loaders.camera_loader import CameraData
from src.render_functions import render_3d_boxes_as_lines_imgproc, get_boxes
from src.traffic_light import get_lightcolors, trafficlightcolor_to_bgr

class Renderer:
    """
    A class for rendering sensor data and corresponding annotations for a given DataItem.
    """
    Color = Tuple[int, int, int]

    def __init__(self, object_type: str):
        self.object_type = object_type
        self.image_resize_ratio = 60

    def render(self, data: DataItem):
        """
        Renders sensor data for a given keyframe.

        Args:
            data: DataItem storing the sensor data and annotations
        """
        if data.camera_data:
            self.render_camera(data.camera_data, data.annotations)

        cv2.waitKey(500)

    def render_camera(self, camera_data: CameraData, annotation: Annotation):
        """
        Renders camera data for a given keyframe.

        Args:
            camera_data: CameraData storing sensor data for each camera
            annotation: Annotation instance, stores annotated traffic light or traffic sign objects
        """
        for camera in camera_data:
            img = camera.image
            camera_params = camera.camera_params

            img = self.plot_image_annotations(img, annotation, camera_params, camera.name)
            img = self.resize_image(camera.name, img)

            cv2.imshow(camera.name, img)

    def plot_image_annotations(self, img: np.array, annotation: Annotation, camera_params: CameraParams,
                               sensor_name: str) -> np.array:
        """
        Visualizes annotations on images corresponding to a given camera.

        Args:
            img: image to plot annotations
            annotation: Annotation instance, stores annotated dynamic objects
            camera_params: stores data required for projections.
            sensor_name: camera name

        Returns:
             img: image with visualized annotations
        """
        thickness = 3
        if self.object_type == "traffic_sign":
            color = (0, 255, 0)
            box_list = get_boxes(annotation.objects, camera_params)
            render_3d_boxes_as_lines_imgproc(img, box_list, camera_params["imgproc_camparams"], color=color, thickness=thickness)
        else:
            # visualize traffic lights
            boxes_tl_by_colors = {}
            for box in annotation.objects:
                light_colors = get_lightcolors(box)
                if light_colors in trafficlightcolor_to_bgr:
                    tl_color = trafficlightcolor_to_bgr[light_colors]
                    boxes_tl_by_colors.setdefault(tl_color, []).append(box)
            for color, boxes_ in boxes_tl_by_colors.items():
                box_list = get_boxes(boxes_, camera_params)
                render_3d_boxes_as_lines_imgproc(img, box_list, camera_params["imgproc_camparams"], color=color, thickness=thickness)

        return img

    def resize_image(self, sensor: str, img: np.array) -> np.array:
        """
        Resizes visualization image based on sensor type.

        Args:
            sensor: sensor name
            img: image to be resized

        Returns:
            img: resized image
        """
        scale_percent = self.image_resize_ratio
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        return img
