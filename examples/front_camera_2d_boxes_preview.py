import argparse
import json
import os
import sys

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.annotation import Annotation
from src.camera_params import CameraParams


CAMERA_NAME = 'F_MIDRANGECAM_C'


def get_3d_box_corners(annotation_object):
    center = np.array([annotation_object[f'BoundingBox3D Origin {ax}'] for ax in ['X', 'Y', 'Z']])
    dims = np.array([annotation_object[f'BoundingBox3D Extent {ax}'] for ax in ['X', 'Y', 'Z']])
    quat = [annotation_object[f'BoundingBox3D Orientation Quat {ax}'] for ax in ['X', 'Y', 'Z', 'W']]
    rotation = quat_xyzw_to_matrix(quat)

    unit_corners = np.array([
        [0.5, 0.5, 0.5],
        [0.5, 0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, -0.5, 0.5],
        [-0.5, 0.5, 0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, -0.5],
        [-0.5, -0.5, 0.5],
    ])

    return (rotation @ (unit_corners * dims).T + center[:, None]).T


def quat_xyzw_to_matrix(quat):
    x, y, z, w = quat
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        return np.eye(3)

    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def body_to_camera(points_body, extrinsic):
    points_hom = np.concatenate([points_body, np.ones((points_body.shape[0], 1))], axis=1)
    return (extrinsic @ points_hom.T)[:3, :].T


def project_corners_to_image(corners_body, camera_params):
    corners_camera = body_to_camera(corners_body, camera_params.extrinsic)

    if camera_params.camera_model == 'mei':
        projected = mei_camera_to_image(corners_camera.T, camera_params).T
        valid_mask = corners_camera[:, 2] > 0
    else:
        projected = pinhole_camera_to_image(corners_camera.T[:, :, None], camera_params).squeeze(-1).T
        valid_mask = (corners_camera[:, 2] > 0) & np.all(projected >= 0, axis=1)

    return projected, valid_mask


def corners_to_bbox(projected_corners, valid_mask, image_shape):
    valid_points = projected_corners[valid_mask]
    if len(valid_points) == 0:
        return None

    height, width = image_shape[:2]
    x_min = int(np.floor(np.clip(valid_points[:, 0].min(), 0, width - 1)))
    y_min = int(np.floor(np.clip(valid_points[:, 1].min(), 0, height - 1)))
    x_max = int(np.ceil(np.clip(valid_points[:, 0].max(), 0, width - 1)))
    y_max = int(np.ceil(np.clip(valid_points[:, 1].max(), 0, height - 1)))

    if x_max <= x_min or y_max <= y_min:
        return None

    return x_min, y_min, x_max, y_max


def draw_labelled_bbox(image, bbox, label):
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_width, text_height = text_size
    text_top = max(0, y_min - text_height - baseline - 4)
    text_bottom = text_top + text_height + baseline + 4
    text_right = min(image.shape[1] - 1, x_min + text_width + 8)

    cv2.rectangle(image, (x_min, text_top), (text_right, text_bottom), (0, 255, 0), thickness=-1)
    cv2.putText(
        image,
        label,
        (x_min + 4, text_bottom - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def is_in_front_fov(annotation_object):
    return annotation_object['BoundingBox3D Origin X'] > 0.5


def get_intrinsics(focal_length, principal_point):
    return np.array([
        [focal_length[0], 0, principal_point[0], 0],
        [0, focal_length[1], principal_point[1], 0],
        [0, 0, 1, 0],
    ])


def read_camera_params(sequence_dir, camera_name):
    calibration_path = os.path.join(sequence_dir, 'sensor', 'calibration', 'calibration.json')
    with open(calibration_path, 'r') as stream:
        calibration = json.load(stream)

    params = calibration[camera_name]
    intrinsic = get_intrinsics(params['focal_length_px'], params['principal_point_px'])
    extrinsic = np.array(params['RT_sensor_from_body'])
    dist_coeffs = np.array(params['distortion_coeffs']) if 'distortion_coeffs' in params else np.array([0., 0., 0., 0., 0.])
    camera_params = CameraParams(intrinsic, extrinsic, dist_coeffs, params['model'])
    if params['model'] == 'mei' and 'xi' in params:
        camera_params.xi = params['xi']

    return camera_params


def load_camera_image(sequence_dir, frame_id, camera_name):
    image_path = os.path.join(
        sequence_dir,
        'sensor',
        'camera',
        camera_name,
        f'{camera_name}_{frame_id}.jpg',
    )
    return cv2.imread(image_path)


def mei_camera_to_image(ray, camera_params):
    xi = camera_params.xi
    k1 = camera_params.dist_coeffs[0]
    k2 = camera_params.dist_coeffs[1]
    p1 = camera_params.dist_coeffs[2]
    p2 = camera_params.dist_coeffs[3]
    k3 = camera_params.dist_coeffs[4] if len(camera_params.dist_coeffs) == 5 else 0

    principal_point_x, principal_point_y = camera_params.principal_point
    focal_length_x, focal_length_y = camera_params.focal_length

    ray_x, ray_y, ray_z = ray
    norm = np.sqrt(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z)
    x = ray_x / norm
    y = ray_y / norm
    z = ray_z / norm + xi
    z[np.abs(z) <= 1e-5] = 1e-5

    x = x / z
    y = y / z

    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    coefficient = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    q_x = x * coefficient + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    q_y = y * coefficient + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y)
    q_x = q_x * focal_length_x + principal_point_x
    q_y = q_y * focal_length_y + principal_point_y

    return np.stack([q_x, q_y], axis=0)


def pinhole_camera_to_image(ray, camera_params):
    ray_x, ray_y, ray_z = ray
    x = ray_x / ray_z
    y = ray_y / ray_z

    if camera_params.dist_coeffs is not None:
        k1 = camera_params.dist_coeffs[0]
        k2 = camera_params.dist_coeffs[1]
        p1 = camera_params.dist_coeffs[2]
        p2 = camera_params.dist_coeffs[3]
        k3 = camera_params.dist_coeffs[4]

        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        coefficient = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        qx = x * coefficient + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        qy = y * coefficient + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y)
    else:
        qx = x
        qy = y

    mask = ((x < -5e-2) & (qx > 1e-5)) | ((x > 5e-2) & (qx < -1e-5)) | \
           ((y < -5e-2) & (qy > 1e-5)) | ((y > 5e-2) & (qy < -1e-5))

    delta = 2e-2
    x2 = x + delta * x
    y2 = y + delta * y
    q2x = x2
    q2y = y2

    if camera_params.dist_coeffs is not None:
        r2 = x2 * x2 + y2 * y2
        r4 = r2 * r2
        r6 = r4 * r2
        coefficient = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        q2x = x2 * coefficient + 2.0 * p1 * x2 * y2 + p2 * (r2 + 2.0 * x2 * x2)
        q2y = y2 * coefficient + 2.0 * p2 * x2 * y2 + p1 * (r2 + 2.0 * y2 * y2)

    mask |= (qx * q2x + qy * q2y) < 0
    mask |= (qx * qx + qy * qy) > (q2x * q2x + q2y * q2y)

    qx = qx * camera_params.focal_length[0] + camera_params.principal_point[0]
    qy = qy * camera_params.focal_length[1] + camera_params.principal_point[1]

    mask |= (qx < 0.0) | (camera_params.principal_point[0] * 2 < qx) | \
            (qy < 0.0) | (camera_params.principal_point[1] * 2 < qy)

    mask = np.invert(mask)

    image_point = np.full((2,) + ray.shape[1:3], -1.0)
    image_point[0, mask] = qx[mask]
    image_point[1, mask] = qy[mask]
    return image_point


def collect_frame_paths(root_dir, split, annotation_type):
    frame_paths = []
    split_dir = os.path.join(root_dir, split)

    for sequence in sorted(os.listdir(split_dir)):
        sequence_dir = os.path.join(split_dir, sequence)
        boxes_dir = os.path.join(sequence_dir, annotation_type, 'box', '3d_body')
        if not os.path.isdir(boxes_dir):
            continue

        for frame_name in sorted(os.listdir(boxes_dir)):
            frame_paths.append(os.path.join(boxes_dir, frame_name))

    return frame_paths


def get_frame_context(frame_path):
    sequence_dir = os.path.sep.join(os.path.normpath(frame_path).split(os.path.sep)[:-4])
    frame_name = os.path.splitext(os.path.basename(frame_path))[0]
    frame_id = frame_name.split('_')[1]
    return sequence_dir, frame_id


def render_preview(frame_paths, frame_indices, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for frame_index in frame_indices:
        frame_path = frame_paths[frame_index]
        sequence_dir, frame_id = get_frame_context(frame_path)
        annotation = Annotation(frame_path)
        image = load_camera_image(sequence_dir, frame_id, CAMERA_NAME)
        camera_params = read_camera_params(sequence_dir, CAMERA_NAME)
        label_count = 0

        for annotation_object in annotation.objects:
            if not is_in_front_fov(annotation_object):
                continue

            corners_body = get_3d_box_corners(annotation_object)
            projected_corners, valid_mask = project_corners_to_image(corners_body, camera_params)
            bbox = corners_to_bbox(projected_corners, valid_mask, image.shape)
            if bbox is None:
                continue

            label = annotation_object['ObjectType']
            draw_labelled_bbox(image, bbox, label)
            label_count += 1

        output_path = os.path.join(output_dir, f'frame_{frame_index:05d}_boxes.jpg')
        cv2.imwrite(output_path, image)
        print(f'saved {output_path} with {label_count} boxes')


def main():
    parser = argparse.ArgumentParser(description='Preview front camera 2D boxes from projected 3D cuboid corners.')
    parser.add_argument('--root-dir', default='data', help='Root dir of aiMotive dataset.')
    parser.add_argument('--split', default='urban', help='Dataset subset, e.g. urban/highway/night/rainy.')
    parser.add_argument('--annotation-type', default='traffic_sign', choices=['traffic_sign', 'traffic_light'], help='Annotation type to render.')
    parser.add_argument('--num-frames', type=int, default=3, help='Number of frames to render.')
    parser.add_argument('--start-index', type=int, default=0, help='Dataset index to start from.')
    parser.add_argument('--output-dir', default='examples/output/front_camera_2d_boxes', help='Directory for preview images.')
    args = parser.parse_args()

    frame_paths = collect_frame_paths(args.root_dir, args.split, args.annotation_type)
    end_index = min(len(frame_paths), args.start_index + args.num_frames)
    frame_indices = range(args.start_index, end_index)
    render_preview(frame_paths, frame_indices, args.output_dir)


if __name__ == '__main__':
    main()
