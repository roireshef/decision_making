from abc import abstractmethod

import numpy as np

from decision_making.src.global_constants import EXP_CLIP_TH
from decision_making.src.planning.types import CartesianTrajectory, C_YAW, CartesianState, C_Y, C_X
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import DynamicObject
from mapping.src.transformations.geometry_utils import CartesianFrame


class SigmoidBoxObstacle:
    def __init__(self, length: float, width: float, k: float, margin: float):
        """
        :param length: length of the box in its own longitudinal axis (box's x)
        :param width: length of the box in its own lateral axis (box's y)
        """
        self._length = length
        self._width = width
        self._k = k
        self._margin = margin

    @property
    def length(self): return self._length

    @property
    def width(self): return self._width

    @property
    def k(self): return self._k

    @property
    def margin(self): return self._margin

    def compute_cost(self, points: np.ndarray) -> np.ndarray:
        """
        Takes a list of points in vehicle's coordinate frame and returns cost of proximity (to self) for each point
        :param points: either a CartesianTrajectory or CartesianTrajectories
        :return: numpy vector of corresponding trajectory-costs
        """
        if len(points.shape) == 2:
            points = np.array([points])

        # add a third value (=1.0) to each point in each trajectory for multiplication with homo-matrix
        ones = np.ones(points.shape[:2])
        points_ext = np.dstack((points, ones))

        points_proj = self.convert_to_obstacle_coordinate_frame(points_ext)[:, :, :(C_Y+1)]

        # subtract from the distances: 1. the box dimensions (height, width) and the margin
        points_offset = np.subtract(points_proj, [self.length / 2 + self.margin, self.width / 2 + self.margin])

        # compute a sigmoid for each dimension [x, y] of each point (in each trajectory)
        logit_costs = np.divide(1.0, (1.0 + np.exp(np.clip(np.multiply(self.k, points_offset), -np.inf, EXP_CLIP_TH))))

        return np.sum(logit_costs[:, :, C_X] * logit_costs[:, :, C_Y], axis=1)

    @abstractmethod
    def convert_to_obstacle_coordinate_frame(self, homo_trajectories_points: np.ndarray):
        """
        Project all points to the box obstacle's own coordinate-frame (for each trajectory).
        Each trajectory-point is multiplied by the appropriate conversion matrix.
        now each record is the [x, y, 1] relative to the box coordinate frame (box-center).
        :param homo_trajectories_points: a numpy tensor (3D) of the shape [t, p, 3] (t trajectories, p points, [x, y, 1]
        in each point). The 1 constant at the 3rd cell of the vectors are for homogeneous matrices multiplication
        :return: a numpy tensor (3D) of the shape [t, p, 3] (t trajectories, p points, [x, y, 1]) relative to box
        coordinate-frame
        """
        pass


class SigmoidDynamicBoxObstacle(SigmoidBoxObstacle):
    def __init__(self, poses: CartesianTrajectory, length: float, width: float, k: float, margin: float):
        """
        :param poses: array of the object's predicted poses, each pose is np.array([x, y, theta, vel])
        :param length: length of the box in its own longitudinal axis (box's x)
        :param width: length of the box in its own lateral axis (box's y)
        """
        super().__init__(length, width, k, margin)
        self._H_inv = np.zeros((poses.shape[0], 3, 3))
        # conversion matrices from global to relative to obstacle
        # TODO: make this more efficient by removing for loop
        for pose_ind in range(poses.shape[0]):
            H = CartesianFrame.homo_matrix_2d(poses[pose_ind, C_YAW], poses[pose_ind, :C_YAW])
            self._H_inv[pose_ind] = np.linalg.inv(H).transpose()

    def convert_to_obstacle_coordinate_frame(self, homo_trajectories_points: np.ndarray):
        return np.abs(np.einsum('ijk, jkl -> ijl', homo_trajectories_points, self._H_inv))

    @classmethod
    def from_object(cls, obj: DynamicObject, k: float, offset: float, time_samples: np.ndarray, predictor: Predictor):
        """
        Additional constructor that takes a ObjectState from the State object and wraps it
        :param obj: ObjectState object from State object (in global coordinates)
        :param k:
        :param offset:
        :param time_samples: [sec] time period for prediction (absolute time)
        :param predictor:
        :return: new instance
        """
        # get predictions of the dynamic object in global coordinates
        predictions = predictor.predict_object(obj, time_samples)
        return cls(predictions, obj.size.length, obj.size.width, k, offset)


class SigmoidStaticBoxObstacle(SigmoidBoxObstacle):
    """
    Static 2D obstacle represented in vehicle's coordinate frame that computes sigmoid costs for
    points in the vehicle's coordinate frame
    """

    # width is on y, length is on x
    def __init__(self, pose: CartesianState, length: float, width: float, k: float, margin: float):
        """
        :param pose: 1D numpy array [x, y, theta, vel] that represents object's pose
        :param length: length of the box in its own longitudinal axis (box's x)
        :param width: length of the box in its own lateral axis (box's y)
        :param k: sigmoid's  exponent coefficient
        :param margin: center of sigmoid offset
        """
        super().__init__(length, width, k, margin)
        H = CartesianFrame.homo_matrix_2d(pose[C_YAW], pose[:C_YAW])
        self._H_inv = np.linalg.inv(H).transpose()

    def convert_to_obstacle_coordinate_frame(self, homo_trajectories_points: np.ndarray):
        return np.abs(np.einsum('ijk, kl -> ijl', homo_trajectories_points, self._H_inv))

    @classmethod
    def from_object(cls, obj: DynamicObject, k: float, offset: float):
        """
        Additional constructor that takes a ObjectState from the State object and wraps it
        :param obj: ObjectState object from State object (in global coordinates)
        :param k:
        :param offset:
        :return: new instance
        """
        return cls(np.array([obj.x, obj.y, obj.yaw, 0]), obj.size.length, obj.size.width, k, offset)
