import os
from typing import List


class Sequence:
    """
    This class represents a sequence (a 15 sec long annotated recording).

    Attributes:
        path: path to the sequence
        path_to_detections: relative path to the traffic light or traffic sign annotation
        keyframes: a list of keyframe paths
    """
    def __init__(self, path: str, object_type: str) :
        """
        Args:
            path: path to the sequence
        """
        assert object_type in ["traffic_light", "traffic_sign"], "The object type must be either traffic_light or traffic_sign!"
        self.path = path
        self.path_to_detections = "traffic_light/box/3d_body" if object_type == "traffic_light" else "traffic_sign/box/3d_body"
        self.keyframes = sorted(os.listdir(os.path.join(path, self.path_to_detections)))

    def get_frames(self) -> List[str]:
        """
        Collects the paths of the keyframes corresponding to the sequence.

        Returns:
            a list of keyframe paths
        """
        return [os.path.join(self.path, self.path_to_detections, keyframe) for keyframe in self.keyframes]

