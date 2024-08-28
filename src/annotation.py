import json

class Annotation:
    """
    This data structure stores the annotated objects for a given keyframe.

    Attributes:
        objects: list of annotated objects, each object is represented as a dict, example object:
        {
            "ActorName": "TRAFFIC_LIGHT 546",
            "ObjectId": 546,
            "BoundingBox3D Origin X": 29.084439949489024,
            "BoundingBox3D Origin Y": -6.630437812302262,
            "BoundingBox3D Origin Z": 3.726494354195893,
            "BoundingBox3D Extent X": 0.7622793551879409,
            "BoundingBox3D Extent Y": 0.7622793551879409,
            "BoundingBox3D Extent Z": 1.383425059018828,
            "BoundingBox3D Orientation Quat W": 0.003842946222369344,
            "BoundingBox3D Orientation Quat X": -0.00035709648962646665,
            "BoundingBox3D Orientation Quat Y": 1.372312826687582e-06,
            "BoundingBox3D Orientation Quat Z": 0.9999925520945373,
            "ObjectType": "TRAFFIC_LIGHT",
            "Occluded": {
                "F_MIDRANGECAM_C": 0.00026658663409762084
            },
            "Truncated": 0,
            "ObjectMeta": {
                "SourceID": 546,
                "SubType": "v-3line",
                "Lights": [
                    {
                        "state": "off",
                        "mask": "full",
                        "color": "unknown"
                    },
                    {
                        "state": "off",
                        "mask": "full",
                        "color": "unknown"
                    },
                    {
                        "state": "on",
                        "mask": "full",
                        "color": "green"
                    }
                ],
                "subject": "vehicles"
            }
        },
    """
    def __init__(self, path: str):
        """
        Args:
            path: path to the annotation json file
        """
        self.path = path
        with open(path, 'r') as gt_file:
            annotations = json.load(gt_file)
            self.objects = annotations['CapturedObjects']
            self.frameid = annotations['FrameId']
