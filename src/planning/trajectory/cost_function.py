from abc import abstractmethod
from typing import List

import numpy as np

import rte.python.profiler as prof
from decision_making.src.global_constants import EXP_CLIP_TH, PLANNING_LOOKAHEAD_DIST
from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams
from decision_making.src.planning.types import C_YAW, CartesianState, C_Y, C_X, \
    CartesianTrajectories, CartesianPaths2D, CartesianPoint2D, C_A, C_K, C_V, CartesianExtendedTrajectories, \
    FrenetTrajectories2D, FS_DX
from decision_making.src.planning.utils.math import Math
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.state.state import State, DynamicObject
from mapping.src.service.map_service import MapService
from mapping.src.transformations.geometry_utils import CartesianFrame


class SigmoidBoxObstacle:
    def __init__(self, length: float, width: float, k: float, margin: CartesianPoint2D):
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

    def compute_cost_per_point(self, points: np.ndarray) -> np.ndarray:
        """
        Takes a list of points in vehicle's coordinate frame and returns cost of proximity (to self) for each point
        :param points: either a CartesianPath2D or CartesianPaths2D
        :return: numpy vector of corresponding costs per point
        """
        if len(points.shape) == 2:
            points = np.array([points])

        points_proj = self.convert_to_obstacle_coordinate_frame(points)

        # subtract from the distances: 1. the box dimensions (height, width) and the margin
        points_offset = np.subtract(points_proj, [self.length / 2 + self.margin[0], self.width / 2 + self.margin[1]])

        # compute a sigmoid for each dimension [x, y] of each point (in each trajectory)
        logit_costs = np.divide(1.0, (1.0 + np.exp(np.clip(np.multiply(self.k, points_offset), -np.inf, EXP_CLIP_TH))))

        return logit_costs[:, :, C_X] * logit_costs[:, :, C_Y]

    def compute_cost(self, points: np.ndarray) -> np.ndarray:
        """
        Takes a list of points in vehicle's coordinate frame and returns cost of proximity (to self) for each point
        :param points: either a CartesianTrajectory or CartesianTrajectories
        :return: numpy vector of corresponding trajectory-costs
        """
        logit_costs = self.compute_cost_per_point(points)
        return np.sum(logit_costs[:, :], axis=1)

    @abstractmethod
    def convert_to_obstacle_coordinate_frame(self, points: CartesianPaths2D) -> CartesianPaths2D:
        """
        Project all points to the box-obstacle's own coordinate-frame (for each trajectory).
        Each trajectory-point is multiplied by the appropriate conversion matrix.
        now each record is relative to the box coordinate frame (box-center).
        :param points: CartesianPaths2D tensor of trajectories in global-frame
        :return: CartesianPaths2D tensor of trajectories in object's-coordinate-frame
        """
        pass


class SigmoidDynamicBoxObstacle(SigmoidBoxObstacle):
    def __init__(self, poses: List[DynamicObject], length: float, width: float, k: float, margin: CartesianPoint2D):
        """
        :param poses: array of the object's predicted poses, each pose is np.array([x, y, theta, vel])
        :param length: length of the box in its own longitudinal axis (box's x)
        :param width: length of the box in its own lateral axis (box's y)
        """
        super().__init__(length, width, k, margin)

        # conversion matrices from global to relative to obstacle
        # TODO: make this more efficient by removing for loop
        self._H_inv = np.zeros((len(poses), 3, 3))
        for pose_ind in range(len(poses)):
            H = CartesianFrame.homo_matrix_2d(poses[pose_ind].cartesian_state[C_YAW], poses[pose_ind].cartesian_state[:C_YAW])
            self._H_inv[pose_ind] = np.linalg.inv(H).transpose()

    def convert_to_obstacle_coordinate_frame(self, points: np.ndarray):
        """ see base method """
        # add a third value (=1.0) to each point in each trajectory for multiplication with homogeneous-matrix
        ones = np.ones(points.shape[:2])
        points_ext = np.dstack((points, ones))

        # this also removes third value (=1.0) from results to return to (x,y) coordinates
        # dimensions - (i) trajectories, (j) timestamp, (k) old-frame-coordinates, (l) new-frame-coordinates
        return np.abs(np.einsum('ijk, jkl -> ijl', points_ext, self._H_inv)[:, :, :(C_Y+1)])

    @classmethod
    def from_object(cls, state: State, obj: DynamicObject, k: float, offset: CartesianPoint2D, time_samples: np.ndarray,
                    predictor: EgoAwarePredictor):
        """
        Additional constructor that takes a ObjectState from the State object and wraps it
        :param state:
        :param obj: ObjectState object from State object (in global coordinates)
        :param k:
        :param offset: longitudinal & lateral margins (half size of ego)
        :param time_samples: [sec] time period for prediction (absolute time)
        :param predictor:
        :return: new instance
        """
        # get predictions of the dynamic object in global coordinates
        predictions = predictor.predict_objects(state, [obj.obj_id], time_samples, None)[obj.obj_id]
        return cls(predictions, obj.size.length, obj.size.width, k, offset)


class SigmoidStaticBoxObstacle(SigmoidBoxObstacle):
    """
    Static 2D obstacle represented in vehicle's coordinate frame that computes sigmoid costs for
    points in the vehicle's coordinate frame
    """

    # width is on y, length is on x
    def __init__(self, pose: CartesianState, length: float, width: float, k: float, margin: CartesianPoint2D):
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

    def convert_to_obstacle_coordinate_frame(self, points: CartesianTrajectories):
        # add a third value (=1.0) to each point in each trajectory for multiplication with homogeneous-matrix
        ones = np.ones(points.shape[:2])
        points_ext = np.dstack((points, ones))

        # this also removes third value (=1.0) from results to return to (x,y) coordinates
        # dimensions - (i) trajectories, (j) timestamp, (k) old-frame-coordinates, (l) new-frame-coordinates
        return np.abs(np.einsum('ijk, kl -> ijl', points_ext, self._H_inv)[:, :, :(C_Y+1)])

    @classmethod
    def from_object(cls, obj: DynamicObject, k: float, offset: CartesianPoint2D):
        """
        Additional constructor that takes a ObjectState from the State object and wraps it
        :param obj: ObjectState object from State object (in global coordinates)
        :param k:
        :param offset:
        :return: new instance
        """
        return cls(np.array([obj.x, obj.y, obj.yaw, 0]), obj.size.length, obj.size.width, k, offset)


import time

class Costs:

    @staticmethod
    def compute_pointwise_costs(ctrajectories: CartesianExtendedTrajectories, ftrajectories: FrenetTrajectories2D,
                                state: State, params: TrajectoryCostParams,
                                global_time_samples: np.ndarray, predictor: EgoAwarePredictor, dt: float) -> \
            [np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute obstacle, deviation and jerk costs for every trajectory point separately.
        It creates a costs tensor of size N x M x 3, where N is trajectories number, M is trajectory length.
        :param ctrajectories: numpy tensor of trajectories in cartesian-frame
        :param ftrajectories: numpy tensor of trajectories in frenet-frame
        :param state: the state object (that includes obstacles, etc.)
        :param params: parameters for the cost function (from behavioral layer)
        :param global_time_samples: [sec] time samples for prediction (global, not relative)
        :param predictor: predictor instance to use to compute future localizations for DyanmicObjects
        :param dt: time step of ctrajectories
        :return: point-wise cost components: obstacles_costs, deviations_costs, jerk_costs.
        The tensor shape: N x M x 3, where N is trajectories number, M is trajectory length.
        """
        ''' OBSTACLES (Sigmoid cost from bounding-box) '''
        # st = time.time()
        # with prof.time_range('old_compute_obstacle_costs{ctrajectories: %s, objects'):
        #     obstacles_costs = Costs.old_compute_obstacle_costs(ctrajectories, state, params, global_time_samples, predictor)
        # print('obs_cost_time=%f' % (time.time()-st))

        # st = time.time()
        obstacles_costs = Costs.compute_obstacle_costs(ctrajectories, state, params, global_time_samples, predictor)
        # print('obs_cost_time=%f' % (time.time()-st))

        ''' DEVIATIONS FROM LANE/SHOULDER/ROAD '''
        deviations_costs = Costs.compute_deviation_costs(ftrajectories, params)

        ''' JERK COST '''
        jerk_costs = Costs.compute_jerk_costs(ctrajectories, params, dt)

        return np.dstack((obstacles_costs, deviations_costs, jerk_costs))

    @staticmethod
    def old_compute_obstacle_costs(ctrajectories: CartesianExtendedTrajectories, state: State, params: TrajectoryCostParams,
                                   global_time_samples: np.ndarray, predictor: EgoAwarePredictor):
        """
        :param ctrajectories: numpy tensor of trajectories in cartesian-frame
        :param state: the state object (that includes obstacles, etc.)
        :param params: parameters for the cost function (from behavioral layer)
        :param global_time_samples: [sec] time samples for prediction (global, not relative)
        :param predictor: predictor instance to use to compute future localizations for DynamicObjects
        :return: MxN matrix of obstacle costs per point, where N is trajectories number, M is trajectory length
        """
        offset = np.array([params.obstacle_cost_x.offset, params.obstacle_cost_y.offset])
        close_obstacles = \
            [SigmoidDynamicBoxObstacle.from_object(state= state, obj=obs, k=params.obstacle_cost_x.k, offset=offset,
                                                   time_samples=global_time_samples, predictor=predictor)
             for obs in state.dynamic_objects
             if np.linalg.norm([obs.x - state.ego_state.x, obs.y - state.ego_state.y]) < PLANNING_LOOKAHEAD_DIST]

        if len(close_obstacles) > 0:
            cost_per_obstacle = [obs.compute_cost_per_point(ctrajectories[:, :, 0:(C_Y+1)]) for obs in close_obstacles]
            obstacles_costs = params.obstacle_cost_x.w * np.sum(cost_per_obstacle, axis=0)
        else:
            obstacles_costs = np.zeros((ctrajectories.shape[0], ctrajectories.shape[1]))
        return obstacles_costs

    @staticmethod
    def compute_obstacle_costs(ctrajectories: CartesianExtendedTrajectories, state: State,
                               params: TrajectoryCostParams, global_time_samples: np.ndarray,
                               predictor: RoadFollowingPredictor):
        """
        :param ctrajectories: numpy tensor of trajectories in cartesian-frame
        :param state: the state object (that includes obstacles, etc.)
        :param params: parameters for the cost function (from behavioral layer)
        :param global_time_samples: [sec] time samples for prediction (global, not relative)
        :param predictor: predictor instance to use to compute future localizations for DyanmicObjects
        :return: MxN matrix of obstacle costs per point, where N is trajectories number, M is trajectory length
        """
        # Filter close objects
        close_objects = [obs for obs in state.dynamic_objects
                   if np.linalg.norm([obs.x - state.ego_state.x, obs.y - state.ego_state.y]) < PLANNING_LOOKAHEAD_DIST]

        with prof.time_range('new_compute_obstacle_costs{objects: %d, ctraj_shape: %s}' % (len(close_objects), ctrajectories.shape)):
            if len(close_objects) == 0:
                return np.zeros((ctrajectories.shape[0], ctrajectories.shape[1]))

            # TODO: this assumes everybody on the same road!
            road_frenet = MapService.get_instance()._rhs_roads_frenet[state.ego_state.map_state.road_id]

            # Predict objects' future movement, then project predicted objects' states to Cartesian frame
            objects_fstates = np.array([obs.map_state.road_fstate for obs in close_objects])
            objects_predicted_ftrajectories = predictor.predict_frenet_states(
                objects_fstates, global_time_samples - state.ego_state.timestamp_in_sec)
            objects_predicted_ctrajectories = road_frenet.ftrajectories_to_ctrajectories(objects_predicted_ftrajectories)

            objects_sizes = np.array([[obs.size.length, obs.size.width] for obs in close_objects])
            ego_size = np.array([state.ego_state.size.length, state.ego_state.size.width])

            # Compute the distance to the closest point in every object to ego's boundaries (on the length and width axes)
            distances = Costs.compute_distances_to_objects(ctrajectories, objects_predicted_ctrajectories, objects_sizes, ego_size)

            # compute a flipped-sigmoid for distances in each dimension [x, y] of each point (in each trajectory)
            k = np.array([params.obstacle_cost_x.k, params.obstacle_cost_y.k])
            offset = np.array([params.obstacle_cost_x.offset, params.obstacle_cost_y.offset])
            points_offset = distances - offset
            per_dimension_cost = np.divide(1.0, (1.0+np.exp(np.minimum(np.multiply(k, points_offset), EXP_CLIP_TH))))

            # multiply dimensional flipped-logistic costs, so that big values are where the two dimensions have
            # negative distance, i.e. collision
            per_point_cost = per_dimension_cost.prod(axis=-1)

            per_trajectory_point_cost = params.obstacle_cost_x.w * np.sum(per_point_cost, axis=1)

            return per_trajectory_point_cost

    @staticmethod
    @prof.ProfileFunction()
    def compute_distances_to_objects(ego_ctrajectories: CartesianExtendedTrajectories,
                                     objects_ctrajectories: CartesianExtendedTrajectories,
                                     objects_sizes: np.array, ego_size: np.array):
        """
        Given M trajectories of ego vehicle of length T timestamps and M objects' predictions of length T timestamps
        each, this functions computes the longitudinal and lateral distances from ego's boundaries to the closest point
        in the objects.
        :param ego_ctrajectories: numpy array of M x CartesianExtendedTrajectory - of M ego's trajectories over time
        :param objects_ctrajectories: numpy array of N x CartesianExtendedTrajectory - of N objects' predictions over time
        :param objects_sizes: numpy array of shape [N, 2] - for each object in N objects - the length and width
        :param ego_size: 1D numpy array of ego's length and width
        :return: numpy array of shape [M, N, T, 2] for M ego's trajectories, N objects, T timestamps,
        distance in ego's [longitudinal, lateral] axes
        """
        objects_H = CartesianFrame.homo_tensor_2d(objects_ctrajectories[:, :, C_YAW],
                                                  objects_ctrajectories[:, :, [C_X, C_Y]])
        objects_H_inv = np.linalg.inv(objects_H)
        objects_H_inv_transposed_trimmed = objects_H_inv.transpose((0, 1, 3, 2))[..., :2]

        ego_points = ego_ctrajectories[:, :, [C_X, C_Y]]
        ego_points_ext = np.dstack((ego_points, np.ones(ego_points.shape[:2])))

        # Ego-center coordinates are projected onto the objects' reference frames [M, N, T, 2]
        # with M ego-trajectories, N objects, T timestamps.
        ego_centers_in_objs_frame = np.einsum('mti, ntij -> mntj', ego_points_ext, objects_H_inv_transposed_trimmed)

        # deduct ego and objects' half-sizes on both dimensions (to reflect objects' boundaries and not center-point)
        distances_from_ego_boundaries = np.abs(ego_centers_in_objs_frame) - 0.5 * (objects_sizes[:, np.newaxis] + ego_size)

        return distances_from_ego_boundaries

    @staticmethod
    def compute_deviation_costs(ftrajectories: FrenetTrajectories2D, params: TrajectoryCostParams):
        """
        Compute point-wise deviation costs from lane, shoulders and road together.
        :param ftrajectories: numpy tensor of trajectories in frenet-frame
        :param params: parameters for the cost function (from behavioral layer)
        :return: MxN matrix of deviation costs per point, where N is trajectories number, M is trajectory length.
        """
        deviations_costs = np.zeros((ftrajectories.shape[0], ftrajectories.shape[1]))

        # add to deviations_costs the costs of deviations from the left [lane, shoulder, road]
        for sig_cost in [params.left_lane_cost, params.left_shoulder_cost, params.left_road_cost]:
            left_offsets = ftrajectories[:, :, FS_DX] - sig_cost.offset
            deviations_costs += Math.clipped_sigmoid(left_offsets, sig_cost.w, sig_cost.k)

        # add to deviations_costs the costs of deviations from the right [lane, shoulder, road]
        for sig_cost in [params.right_lane_cost, params.right_shoulder_cost, params.right_road_cost]:
            right_offsets = np.negative(ftrajectories[:, :, FS_DX]) - sig_cost.offset
            deviations_costs += Math.clipped_sigmoid(right_offsets, sig_cost.w, sig_cost.k)
        return deviations_costs

    @staticmethod
    def compute_jerk_costs(ctrajectories: CartesianExtendedTrajectories, params: TrajectoryCostParams, dt: float) -> \
            np.array:
        """
        Compute point-wise jerk costs as weighted sum of longitudinal and lateral jerks
        :param ctrajectories: numpy tensor of trajectories in cartesian-frame
        :param params: parameters for the cost function (from behavioral layer)
        :param dt: time step
        :return: MxN matrix of jerk costs per point, where N is trajectories number, M is trajectory length.
        """
        lon_jerks, lat_jerks = Jerk.compute_jerks(ctrajectories, dt)
        jerk_costs = params.lon_jerk_cost * lon_jerks + params.lat_jerk_cost * lat_jerks
        return np.c_[np.zeros(jerk_costs.shape[0]), jerk_costs]


class Jerk:
    @staticmethod
    def compute_jerks(ctrajectories: CartesianExtendedTrajectories, dt: float):
        """
        Compute longitudinal and lateral jerks based on cartesian trajectories.
        :param ctrajectories: array[trajectories_num, timesteps_num, 6] of cartesian trajectories
        :param dt: time step for acceleration derivative by time
        :return: two ndarrays of size ctrajectories.shape[0]. Longitudinal and lateral jerks for all ctrajectories
        """
        # divide by dt^2 (squared a_dot) and multiply by dt (in the integral)
        lon_jerks = np.square(np.diff(ctrajectories[:, :, C_A], axis=1)) / dt
        lat_jerks = np.square(np.diff(ctrajectories[:, :, C_K] * np.square(ctrajectories[:, :, C_V]), axis=1)) / dt
        return lon_jerks, lat_jerks

