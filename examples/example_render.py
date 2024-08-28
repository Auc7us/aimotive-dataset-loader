import argparse

from src.aimotive_dataset import AiMotiveDataset
from src.renderer import Renderer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example script for visualizing aiMotive Traffic Sign/Traffic Light Dataset.')

    parser.add_argument("--root-dir", default="data",
                        type=str, help="Root dir of aiMotive TL/TS Dataset.")
    parser.add_argument("--object-type", default="traffic_sign",
                        type=str, help="Object type. Options: [traffic_sign, traffic_light]")
    args = parser.parse_args()

    train_dataset = AiMotiveDataset(args.root_dir, args.object_type)
    renderer = Renderer(train_dataset.object_type)
    for data in train_dataset:
        renderer.render(data)
