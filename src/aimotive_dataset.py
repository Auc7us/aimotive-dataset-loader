import os

from typing import List

from src.data_loader import DataLoader, DataItem
from src.sequence import Sequence


class AiMotiveDataset:
    """
    AiMotive's traffic light/traffic sign dataset.
    The dataset consists of four cameras and corresponding
    3D bounding box annotations of traffic light or traffic sign objects.

    Attributes:
        object_type: the type of object, that can be either traffic_light or traffic_sign
        dataset_index: a list of keyframe paths
        data_loader: a DataLoader class for loading sensor data.
    """
    def __init__(self, root_dir: str, object_type: str):
        """
        Args:
            root_dir: path to the dataset
        """
        assert object_type in ["traffic_light", "traffic_sign"], "The object type must be either traffic_light or traffic_sign!"
        self.object_type = object_type
        self.dataset_index = self.get_frames(root_dir)
        self.data_loader = DataLoader(self.dataset_index)

    def __len__(self):
        return len(self.dataset_index)

    def __getitem__(self, index: int) -> DataItem:
        return self.data_loader[self.dataset_index[index]]

    def get_frames(self, path: str) -> List[str]:
        """
        Collects the keyframe paths.

        Args:
            path: path to the dataset

        Returns:
            data_paths: a list of keyframe paths

        """
        data_paths = []
        odd_path = path
        for odd in sorted(os.listdir(odd_path)):
            for seq in sorted(os.listdir(os.path.join(path, odd))):
                seq_path = os.path.join(odd_path, odd, seq)
                sequence = Sequence(seq_path, self.object_type)
                data_paths.extend(sequence.get_frames())

        return data_paths


if __name__ == '__main__':
    root_directory = "../data"
    dataset = AiMotiveDataset(root_directory, object_type='traffic_light')
    for data in dataset:
        print(data.annotations.path)

