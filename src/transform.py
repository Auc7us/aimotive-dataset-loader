import quaternion
import numpy as np

class Transform:
    '''
    Transform consists of a Rotation (as a quaternion), and a Translation (position).
    Beware: while most RT matrices can be described using a quaternion and a translation,
    the ones which change handedness (left handed <-> right handed) are going to fail,
    (for example: a quaternion cant  describe a 'rotation' that flips the x axis)
    so you best keep those in matrix form until you convert handedness.
    Otherwise you could swap a rotation with axis_swap.
    '''

    def __init__(self, rotation_quat: quaternion.quaternion = None, translation: np.array = None):
        assert type(translation) != quaternion.quaternion
        if rotation_quat is None:
            rotation_quat = quaternion.one
        if translation is None:
            translation = np.array([0, 0, 0], dtype=np.float32)
        if type(rotation_quat) in [list, tuple, np.ndarray]:
            rotation_quat = quaternion.quaternion(*rotation_quat)
        assert type(rotation_quat) == quaternion.quaternion
        self.rotation_quat = rotation_quat
        assert len(translation) == 3
        self.translation = np.array(translation)

    @staticmethod
    def matmul(left: 'Transform', right: 'Transform') -> 'Transform':
        return left * right

    def mul(self, other: 'Transform') -> 'Transform':
        return self * other

    def __matmul__(self, other: 'Transform') -> 'Transform':
        return self * other

    def __mul__(self, other: 'Transform') -> 'Transform':
        pos = quaternion.rotate_vectors(other.rotation_quat, self.translation) + other.translation
        q = other.rotation_quat * self.rotation_quat
        t = Transform(q, pos)
        return t

    def inv(self) -> 'Transform':
        q = self.rotation_quat.inverse()
        pos = -quaternion.rotate_vectors(q, self.translation)
        t = Transform(q, pos)
        return t

    @staticmethod
    def fromMatrix(RT_mat_postmul) -> 'Transform':
        RT = RT_mat_postmul
        if np.linalg.det(RT[:3, :3]) < 0:
            raise Exception("Can't represent improper rotations (R matrix determinant = -1)")
        t = Transform(rotation_quat=quaternion.from_rotation_matrix(RT[:3, :3].T, nonorthogonal=False),
                      translation=RT[3, :3])
        return t

    def to_RT_Matrix(self):
        mx = np.identity(4, dtype=np.float32)
        mx[:3, :3] = quaternion.as_rotation_matrix(self.rotation_quat).T
        mx[3, :3] = self.translation
        return mx

    @staticmethod
    def RT_inverse(RT_mat_postmul: np.ndarray) -> np.ndarray:
        R = RT_mat_postmul[:3, :3]
        T = RT_mat_postmul[3, :3]
        mx = np.identity(4, dtype=np.float32)
        mx[:3, :3] = R.T
        mx[3, :3] = -T @ R.T
        return mx

    def axis_swap(self, A: np.ndarray) -> 'Transform':
        q = self.rotation_quat.copy()
        pos = self.translation @ A
        sign = 1 if np.linalg.det(A) > 0 else -1
        q.vec = np.array(q.vec @ A) * sign
        t = Transform(q, pos)

        return t

    # Transforms point with the self Transform. (Rotation and Translation).
    def transformPoint(self, point):
        return self.transformPoints([point])[0]

    def transformPoints(self, points):
        points = np.array(points)
        assert points.shape[1] in [3, 4]
        points_rotated = quaternion.rotate_vectors(self.rotation_quat, points[:, :3])
        points_translated = points_rotated + self.translation
        if points.shape[1] == 4:
            points_translated = np.c_[points_translated, np.ones((points.shape[0], 1))]
        return points_translated

    def __rmul__(self, points: np.ndarray) -> np.ndarray:
        return points @ self

    def __rmatmul__(self, points: np.ndarray) -> np.ndarray:
        points = np.array(points)
        if points.ndim == 1:
            return self.transformPoint(points)
        return self.transformPoints(points)

    def __repr__(self):
        return "Transform(quaternion({:.5f}, {:.3f}, {:.3f}, {:.3f}), [{:.2f}, {:.2f}, {:.2f}])".format( \
            self.rotation_quat.w, self.rotation_quat.x, self.rotation_quat.y, self.rotation_quat.z, \
            self.translation[0], self.translation[1], self.translation[2])
