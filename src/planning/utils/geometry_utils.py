import numpy as np
from src.planning.global_constants import *

class CartesianFrame:
    @staticmethod
    def homo_matrix_2d(rotation_angle: float, translation: np.ndarray) -> np.ndarray:
        """
        Generates a 2D homogeneous matrix for cartesian frame projections
        :param rotation_angle: yaw in radians
        :param translation: a [x, y] translation vector
        :return: a 3x3 numpy matrix for projection (translation+rotation)
        """
        pass

    @staticmethod
    def homo_matrix_3d(rotation_angle: float, translation: np.ndarray) -> np.ndarray:
        """
        Generates a 2D homogeneous matrix for cartesian frame projections
        :param rotation_angle: yaw in radians
        :param translation: a [x, y] translation vector
        :return: a 3x3 numpy matrix for projection (translation+rotation)
        """
        pass

    @staticmethod
    def homo_matrix_3d_from_quaternion(quaternion: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """
        Generates a 2D homogeneous matrix for cartesian frame projections
        :param quaternion: numpy array of quaternion rotation
        :param translation: a [x, y] translation vector
        :return: a 3x3 numpy matrix for projection (translation+rotation)
        """
        pass

    @staticmethod
    def add_yaw_and_curvature(xy_points: np.ndarray) -> np.ndarray:
        """
        Takes a matrix of curve points ([x, y] only) and adds yaw and curvature columns, by computing pseudo-derivatives
        :param xy_points: a numpy matrix of shape [n, 2]
        :return: a numpy matrix of shape [n, 4]
        """
        pass

class FrenetMovingFrame:
    """
    A 2D Frenet moving coordinate frame. Within this class: fpoint, fstate and ftrajectory are in frenet frame;
    cpoint, cstate, and ctrajectory are in cartesian coordinate frame
    """
    def __init__(self, curve_xy, resolution=TRAJECTORY_ARCLEN_RESOLUTION):
        self.__curve = CartesianFrame.add_yaw_and_curvature(curve_xy)
        self.__ds = resolution
        self.__h_tensor = np.array([CartesianFrame.homo_matrix_2d(self.__curve[s_idx, 2], self.__curve[s_idx, 0:2])
                                    for s_idx in range(len(self.__curve))])

    def get_homo_matrix_2d(self, s_idx: float) -> np.ndarray:
        """
        Returns the homogeneuos matrix (rotation+translation) for the FrenetFrame at a point along the curve
        :param s_idx: distance travelled from the beginning of the curve (in self.__ds units)
        :return: numpy array of shape [3,3] of homogeneous matrix
        """
        if self.__h_tensor.size > s_idx:
            return self.__h_tensor[s_idx]
        else:
            raise ValueError('index ' + str(s_idx) + 'is not found in __h_tensor (probably __h_tensor is not cached)')

    def cpoint_to_fpoint(self, cpoint: np.ndarray) -> np.ndarray:
        """
        Transforms cartesian-frame point [x, y] to frenet-frame point (using self.curve) \n
        :param cpoint: cartesian coordinate frame state-vector [x, y, ...]
        :return: numpy vector of FrenetPoint instance [sx, dx]
        """
        pass

    def fpoint_to_cpoint(self, fpoint: np.ndarray) -> np.ndarray:
        """
        Transforms frenet-frame point to cartesian-frame point (using self.curve) \n
        :param fpoint: FrenetPoint instance (relative to self.curve moving frenet-frame)
        :return: cartesian-frame point [x, y]
        """
        pass

    # currently this is implemented. We should implement the next method and make this one wrap a single trajectory
    # and send it to the next method.
    def ftrajectory_to_ctrajectory(self, ftrajectory: np.ndarray) -> np.ndarray:
        """
        Transforms Frenet-frame trajectory to cartesian-frame trajectory, using tensor operations
        :param ftrajectory: a numpy matrix of rows of the form [sx, sv, sa, dx, dv, da]
        :return: a numpy matrix of rows of the form [x, y, theta, v, a, k] in car's coordinate frame
        """
        pass

    def ftrajectories_to_ctrajectories(self, ftrajectories: np.ndarray) -> np.ndarray:
        """
        Transforms Frenet-frame trajectories to cartesian-frame trajectories, using tensor operations
        :param ftrajectories: a numpy tensor with dimensions [0 - trajectories, 1 - trajectory points,
            2 - frenet-state [sx, sv, sa, dx, dv, da])
        :return: a numpy tensor with dimensions [0 - trajectories, 1 - trajectory points,
            2 - cartesian-state [x, y, theta, v, a, k]) in car's coordinate frame
        """
        pass
