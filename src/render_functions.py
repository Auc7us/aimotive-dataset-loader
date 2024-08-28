import numpy as np
import quaternion
import cv2

from src.transform import Transform

box3d_corners_base = np.array([[1, -1, +1, -1, +1, -1, +1, -1],
                               [1, +1, -1, -1, +1, +1, -1, -1],
                               [1, +1, +1, +1, -1, -1, -1, -1]], dtype=np.float32).T

box3d_corners_ext = np.vstack((box3d_corners_base, np.array([[0, 1, 0], [0, 1, 1]], dtype=np.float32)))

def get_box3d_corners_ext(box3d):
    return get_box3d_corners_transformed(box3d, np.copy(box3d_corners_ext))

def get_box3d_corners_transformed(box3d, corners):
    corners *= box3d.size * 0.5
    corners = quaternion.rotate_vectors(box3d.ori_quat, corners)
    corners += box3d.pos
    return corners

def get_box3d_corners(box3d):
    return get_box3d_corners_transformed(box3d, np.copy(box3d_corners_base))

class Box3d:
    def __init__(self, pos, size, ori_quat: quaternion.quaternion, object_id = None, object_type = None, object_meta = None):
        self.pos = np.array(pos)
        self.size = np.array(size)
        self.ori_quat = ori_quat
        self.object_id = object_id
        self.object_type = object_type
        self.object_meta = object_meta

def get_3d_bbox_corners_projections(box, camera):
    rot_quat = quaternion.quaternion(box["BoundingBox3D Orientation Quat W"],
                                     box["BoundingBox3D Orientation Quat X"],
                                     box["BoundingBox3D Orientation Quat Y"],
                                     box["BoundingBox3D Orientation Quat Z"])

    pos = box["BoundingBox3D Origin X"], box["BoundingBox3D Origin Y"], box["BoundingBox3D Origin Z"]
    sizes = box['BoundingBox3D Extent X'], box['BoundingBox3D Extent Y'], box['BoundingBox3D Extent Z']

    width, height, length = sizes
    newbox = Box3d(pos, size=[width, height, length], ori_quat=rot_quat, object_id=box["ObjectId"],
                   object_type=box.get("ObjectType"), object_meta=box.get("ObjectMeta"))

    box_corners = get_box3d_corners(newbox)
    box_corners_ = np.concatenate([box_corners, np.ones((box_corners.shape[0], 1))], axis=1)
    box_corners_view = (box_corners_ @ camera["body_to_view"])[:, :3]
    box_corners_view_tmp = box_corners_view.reshape((int(box_corners_view.shape[0] / 8), 8, 3))
    box_center_view = box_corners_view_tmp.mean(axis=1)
    box_center_view /= np.linalg.norm(box_center_view, axis=1, keepdims=True)
    points, msk = get_projected_pts_with_mask_imgproc(box_corners_view, camera["imgproc_camparams"],
                                                                 camera["image_resolution_px"][0],
                                                                 camera["image_resolution_px"][1])
    return points, msk

def get_2d_bbox_of_3d_bbox(box, camera):
    points, msk = get_3d_bbox_corners_projections(box, camera)
    if len(msk[msk]) == 8:
        tl_x, tl_y, br_x, br_y = np.min(points[:,0]), np.min(points[:,1]), np.max(points[:,0]), np.max(points[:,1])
        return [tl_x, tl_y, br_x, br_y]
    else:
        return [-1, -1, -1, -1]

def get_projected_pts_with_mask_imgproc(points, cameraParams, w, h):
    projected_pts = ray_to_image(points.T, cameraParams).T

    # filter out of bounds pts
    in_viewport_mask = np.min(projected_pts >= 0, axis=1) & np.min(projected_pts < (w, h), axis=1)
    projected_pts = projected_pts[in_viewport_mask]

    all_pts_in_viewport = points[in_viewport_mask]
    unprojected_pts = image_to_ray(projected_pts.T, cameraParams).T

    # dot product of pts and their projected+unprojected counterparts
    dot = np.sum(normalized(all_pts_in_viewport) * normalized(unprojected_pts), axis=-1)
    # dot close to 1 means same direction
    pts_projectable = dot > 0.999

    projected_pts = projected_pts[pts_projectable]

    mask = np.copy(in_viewport_mask)
    mask[mask] = pts_projectable
    return projected_pts, mask

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def ray_to_image(ray, parameters):
    if parameters.model == "opencv_pinhole":
        ray = pinhole_view_to_image(ray=ray,
                                    parameters=parameters)
    elif parameters.model == "mei":
        ray = mei_view_to_image(ray=ray,
                                parameters=parameters)
    else:
        raise ValueError
    return ray

def pinhole_view_to_image(parameters, ray):
    x = ray[0] / ray[2]
    y = ray[1] / ray[2]

    def distortPoint(x, y):
        if not parameters.distortion_coeffs is None:
            k1 = parameters.distortion_coeffs[0]
            k2 = parameters.distortion_coeffs[1]
            p1 = parameters.distortion_coeffs[2]
            p2 = parameters.distortion_coeffs[3]
            k3 = parameters.distortion_coeffs[4]

            r2 = x * x + y * y
            r4 = r2 * r2
            r6 = r4 * r2

            coefficient = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

            qx = x * coefficient + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            qy = y * coefficient + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y)
        else:
            qx = x
            qy = y
        return qx, qy

    qx, qy = distortPoint(x, y)
    mask = ((x < -5e-2) & (qx > 1e-5)) | ((x > 5e-2) & (qx < -1e-5)) | \
           ((y < -5e-2) & (qy > 1e-5)) | ((y > 5e-2) & (qy < -1e-5))

    delta = 2e-2
    x2 = x + delta * x
    y2 = y + delta * y
    q2x, q2y = distortPoint(x2, y2)

    mask |= (qx * q2x + qy * q2y) < 0
    mask |= (qx * qx + qy * qy) > (q2x * q2x + q2y * q2y)

    qx = qx * parameters.focal_length_px[0] + parameters.principal_point_px[0]
    qy = qy * parameters.focal_length_px[1] + parameters.principal_point_px[1]

    # viewport check
    mask |= (qx < 0.0) | (parameters.image_resolution_px[0] < qx) | \
            (qy < 0.0) | (parameters.image_resolution_px[1] < qy)

    mask = np.invert(mask)

    # for invalid points we get (-1.0, -1.0) values
    image_point = np.full((2,) + ray.shape[1:3], -1.0)
    image_point[0, mask] = qx[mask]
    image_point[1, mask] = qy[mask]

    return image_point

def mei_view_to_image(parameters, ray):
    xi = parameters.xi
    k1 = parameters.distortion_coeffs[0]
    k2 = parameters.distortion_coeffs[1]
    p1 = parameters.distortion_coeffs[2]
    p2 = parameters.distortion_coeffs[3]
    if len(parameters.distortion_coeffs) == 5:
        k3 = parameters.distortion_coeffs[4]
    else:
        k3 = 0

    x, y, z = ray
    def distort(x, y, z):
        norm = np.sqrt(x * x + y * y + z * z)
        x = x / norm
        y = y / norm
        z = z / norm + xi
        z[np.abs(z) <= 1e-5] = 1e-5

        x = x / z
        y = y / z

        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        coefficient = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        q_x = x * coefficient + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        q_y = y * coefficient + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y)
        return q_x, q_y

    qx, qy = distort(x,y,z)
    mask = ((x < -5e-2) & (qx > 1e-5)) | ((x > 5e-2) & (qx < -1e-5)) | \
           ((y < -5e-2) & (qy > 1e-5)) | ((y > 5e-2) & (qy < -1e-5))

    delta = 2e-2
    x2 = x + delta * x
    y2 = y + delta * y
    q2x, q2y = distort(x2, y2, z)

    mask |= (qx * q2x + qy * q2y) < 0
    mask[z >= 0] |= ((qx * qx + qy * qy) > (q2x * q2x + q2y * q2y))[z >= 0]
    mask[z < 0] |= ((qx * qx + qy * qy) < (q2x * q2x + q2y * q2y))[z < 0]

    qx = qx * parameters.focal_length_px[0] + parameters.principal_point_px[0]
    qy = qy * parameters.focal_length_px[1] + parameters.principal_point_px[1]

    # viewport check
    mask |= (qx < 0.0) | (parameters.image_resolution_px[0] < qx) | \
            (qy < 0.0) | (parameters.image_resolution_px[1] < qy)

    mask = np.invert(mask)

    # for invalid points we get (-1.0, -1.0) values
    image_point = np.full((2,) + ray.shape[1:3], -1.0)
    image_point[0, mask] = qx[mask]
    image_point[1, mask] = qy[mask]
    return image_point

def image_to_ray(image_pts, parameters):
    if parameters.model == "opencv_pinhole":
        ray = pinhole_image_to_view(image_point=image_pts,
                                    parameters=parameters)
    elif parameters.model == "mei":
        ray = mei_image_to_view(image_point=image_pts,
                                parameters=parameters)
    else:
        raise ValueError

    return ray

def pinhole_image_to_view(parameters, image_point):
    distorted_image_point = np.zeros(image_point.shape)
    distorted_image_point[0] = (image_point[0] - parameters.principal_point_px[0]) / \
                               parameters.focal_length_px[0]
    distorted_image_point[1] = (image_point[1] - parameters.principal_point_px[1]) / \
                               parameters.focal_length_px[1]

    k1 = k2 = p1 = p2 = k3 = 0.0
    if parameters.distortion_coeffs is not None:
        k1 = parameters.distortion_coeffs[0]
        k2 = parameters.distortion_coeffs[1]
        p1 = parameters.distortion_coeffs[2]
        p2 = parameters.distortion_coeffs[3]
        k3 = parameters.distortion_coeffs[4]

    undistorted_image_point = np.copy(distorted_image_point)
    undistorted = k1 == 0.0 and k2 == 0.0 and p1 == 0.0 and p2 == 0.0 and k3 == 0.0
    if not undistorted:
        n_iterations = 20
        for i in range(n_iterations):
            xx = undistorted_image_point[0] * undistorted_image_point[0]
            yy = undistorted_image_point[1] * undistorted_image_point[1]
            r2 = xx + yy
            _2xy = 2.0 * undistorted_image_point[0] * undistorted_image_point[1]
            radial_distortion = 1.0 + (k1 + (k2 + k3 * r2) * r2) * r2
            tangential_distortion_X = np.array(p1 * _2xy + p2 * (r2 + 2.0 * xx))
            tangential_distortion_Y = np.array(p1 * (r2 + 2.0 * yy) + p2 * _2xy)

            undistorted_image_point[0] = (distorted_image_point[0] - tangential_distortion_X) / radial_distortion
            undistorted_image_point[1] = (distorted_image_point[1] - tangential_distortion_Y) / radial_distortion

    norm = np.sqrt(undistorted_image_point[0] * undistorted_image_point[0] + undistorted_image_point[1] *
                   undistorted_image_point[1] + 1.0)
    ray = np.zeros((3,) + image_point.shape[1:])
    ray[0] = undistorted_image_point[0] / norm
    ray[1] = undistorted_image_point[1] / norm
    ray[2] = 1.0 / norm

    return ray

def mei_image_to_view(parameters, image_point):
    "https://gerrit.aimotive.com/plugins/gitiles/aimimgproc/+/refs/heads/master/aimcmt/devinc/aimdev/cmt/cameramodeltransf_impl.hpp#274"
    xi = parameters.xi
    k1 = parameters.distortion_coeffs[0]
    k2 = parameters.distortion_coeffs[1]
    p1 = parameters.distortion_coeffs[2]
    p2 = parameters.distortion_coeffs[3]
    k3 = parameters.distortion_coeffs[4]
    principal_point = parameters.principal_point_px
    focal_length = parameters.focal_length_px

    def undist(image_point):
        distorted_point_x = (image_point[0] - principal_point[0]) / focal_length[0]
        distorted_point_y = (image_point[1] - principal_point[1]) / focal_length[1]
        undistorted_point_x = distorted_point_x
        undistorted_point_y = distorted_point_y
        c_num_iterations = 20
        for i in range(c_num_iterations):
            xx = undistorted_point_x * undistorted_point_x
            yy = undistorted_point_y * undistorted_point_y
            r2 = xx + yy
            _2xy = 2.0 * undistorted_point_x * undistorted_point_y
            radial_distortion = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2

            tangential_distortion_x = p1 * _2xy + p2 * (r2 + 2.0 * xx)
            tangential_distortion_y = p1 * (r2 + 2.0 * yy) + p2 * _2xy
            undistorted_point_x = (distorted_point_x - tangential_distortion_x) / radial_distortion
            undistorted_point_y = (distorted_point_y - tangential_distortion_y) / radial_distortion

        return np.stack([undistorted_point_x, undistorted_point_y], axis=0)

    ud_pt = undist(image_point)
    # project to unit sphere - OpenCV Omnidir implementation
    r2 = ud_pt[0] * ud_pt[0] + ud_pt[1] * ud_pt[1]
    a = (r2 + 1.0)
    b = 2 * xi * r2
    cc = r2 * xi * xi - 1

    Disc = b * b - 4 * a * cc
    mask = Disc >= 0.0
    a = a[mask]
    b = b[mask]
    cc = cc[mask]
    Disc = Disc[mask]
    Zs = (-b + np.sqrt(Disc)) / (2 * a)

    ray = np.zeros((3,) + image_point.shape[1:])
    ray[0:2, mask] = ud_pt[0:2, mask] * (Zs + xi)
    ray[2, mask] = Zs

    maskinv = np.invert(mask)
    ray[0:2, maskinv] = 0.0
    ray[2, maskinv] = -1
    return ray

def slice_lines(lines, line_segment_length):
    length = np.linalg.norm(lines[:,0] - lines[:,1], axis=1)
    slice_cnt = np.ceil(length / line_segment_length).astype(np.int32)
    all_lines = [np.zeros([0, 3], dtype=np.float64)]
    for idx, (pt0, pt1) in enumerate(lines):
        line_pts = np.linspace(pt0, pt1, slice_cnt[idx] + 1)
        line_segments = np.repeat(line_pts, 2, axis=0)[1:-1]
        all_lines.append(line_segments)
    all_lines = np.vstack(all_lines).reshape(-1, 2, 3)

    return all_lines

def get_projected_lines_with_mask_imgproc(lines, cameraParams, w, h):
    projected_lines = ray_to_image(lines.T, cameraParams).T

    # filter out of bounds lines
    in_viewport_mask = np.min(projected_lines >= 0, axis=(1, 2)) & np.min(projected_lines < (w, h), axis=(1, 2))
    projected_lines = projected_lines[in_viewport_mask]

    all_lines_in_viewport = lines[in_viewport_mask]
    unprojected_lines = image_to_ray(projected_lines.T, cameraParams).T

    # dot product of lines and their projected+unprojected counterparts
    dot = np.sum(normalized(all_lines_in_viewport) * normalized(unprojected_lines), axis=-1)
    # dot close to 1 means same direction
    line_projectable = np.min(dot, axis=-1) > 0.999

    projected_lines = projected_lines[line_projectable]
    mask = np.copy(in_viewport_mask)
    mask[mask == True] = line_projectable
    return projected_lines, line_projectable

def render_lines_imgproc(img, lines, cameraParams, color=(255, 0, 0), thickness=1, line_segment_length = 0.05):
    lines = lines.reshape(-1, 2, 3)
    w, h = img.shape[1], img.shape[0]

    if line_segment_length == -1:
        all_lines = lines
    else:
        all_lines = slice_lines(lines, line_segment_length)

    if len(all_lines) == 0:
        return

    if cameraParams.model == 'opencv_pinhole':
        all_lines = all_lines[(all_lines[:, 0, 2] > 0) & (all_lines[:, 1, 2] > 0)]
        if len(all_lines) == 0:
            return

    projected_lines, projected_lines_mask = get_projected_lines_with_mask_imgproc(all_lines, cameraParams, w, h)
    projected_lines = projected_lines.astype(np.int64)
    cv2.polylines(img, projected_lines, isClosed=False, color=color, thickness=thickness)

box3d_line_idx = [0, 1, 0, 2, 0, 4,
                  6, 7, 6, 2, 6, 4,
                  5, 4, 5, 1, 5, 7,
                  3, 2, 3, 1, 3, 7]

box3d_line_idx_ext = box3d_line_idx + [8, 9]

def render_3d_boxes_as_lines_imgproc(img, boxes, cameraParams, color=(100, 230, 20), thickness=1,
                                     draw_ids=True, id_offset=(10, 0), fontColor=(200, 200, 250),
                                     fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontThickness=1,
                                     draw_occlusions=False):
    if len(boxes) == 0:
        return
    lines_all = []
    corners_all = []

    id_positions = []
    ids = []
    for box3d in boxes:
        corners = get_box3d_corners_ext(box3d)
        corners_all.append(corners[:8])
        lines = corners[box3d_line_idx_ext]
        lines_all.append(lines)

        id_positions.append(corners[0])
        ids.append(box3d.object_id)
    lines_all = np.array(lines_all)
    render_lines_imgproc(img, lines_all, cameraParams, color=color, thickness=thickness)

    if draw_ids or draw_occlusions:
        id_positions = np.array(id_positions)
        id_positions_proj, id_positions_proj_mask = get_projected_pts_with_mask_imgproc(id_positions, cameraParams,
                                                                                        img.shape[1], img.shape[0])
        id_positions = id_positions[id_positions_proj_mask]
        ids = np.array(ids)[id_positions_proj_mask]
        for id_position, id_position_proj, id in zip(id_positions, id_positions_proj, ids):
            x = np.linalg.norm(id_position)
            fontScale = np.maximum((1 - x / 200), 0) ** 2 * 2
            if fontScale == 0.0:
                continue
            u, v = id_position_proj.astype(int)[:2] + np.array(id_offset)
            u, v = int(u), int(v)

            text = f"{id}" if draw_ids else ""
            cv2.putText(img, org=(u, v), text=text, color=fontColor,
                        fontFace=fontFace, fontScale=fontScale, thickness=fontThickness, lineType=cv2.LINE_AA)

def get_boxes(dict_boxes, camera):
    A_cam_to_view = np.array([[0, 0, 1],
                              [-1, 0, 0],
                              [0, -1, 0]])

    trafo_cam2body = camera['camera_transform']

    newboxes = []
    for box in dict_boxes:
        rot_quat = quaternion.quaternion(box["BoundingBox3D Orientation Quat W"],
                                            box["BoundingBox3D Orientation Quat X"],
                                            box["BoundingBox3D Orientation Quat Y"],
                                            box["BoundingBox3D Orientation Quat Z"])

        pos = box["BoundingBox3D Origin X"], box["BoundingBox3D Origin Y"], box["BoundingBox3D Origin Z"]
        sizes = box['BoundingBox3D Extent X'], box['BoundingBox3D Extent Y'], box['BoundingBox3D Extent Z']

        trafo_body = Transform(rot_quat, pos)
        trafo_view = (trafo_body * trafo_cam2body.inv()).axis_swap(A_cam_to_view)
        pos = trafo_view.translation
        rot_quat = trafo_view.rotation_quat
        sizes = np.abs(sizes @ A_cam_to_view)

        width, height, length = sizes
        newbox = Box3d(pos, size=[width, height, length], ori_quat=rot_quat, object_id=box["ObjectId"],
                        object_type=box.get("ObjectType"), object_meta=box.get("ObjectMeta"))

        newboxes.append(newbox)
    return newboxes
