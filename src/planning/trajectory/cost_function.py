import numpy as np

import rte.python.profiler as prof
from decision_making.src.global_constants import EXP_CLIP_TH, PLANNING_LOOKAHEAD_DIST, SAFETY_SIGMOID_COST, \
    LON_SAFETY_SIGMOID_K_PARAM, LAT_SAFETY_SIGMOID_K_PARAM
from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams
from decision_making.src.planning.types import C_YAW, C_Y, C_X, C_A, C_K, C_V, CartesianExtendedTrajectories, \
    FrenetTrajectories2D, FS_DX
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import State
from mapping.src.transformations.geometry_utils import CartesianFrame


class TrajectoryPlannerCosts:

    @staticmethod
    def compute_pointwise_costs(ctrajectories: CartesianExtendedTrajectories, ftrajectories: FrenetTrajectories2D,
                                state: State, params: TrajectoryCostParams,
                                global_time_samples: np.ndarray, predictor: EgoAwarePredictor, dt: float,
                                reference_route: FrenetSerret2DFrame, safe_distances: np.array) -> \
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
        :param reference_route: GFF reference route
        :param safe_distances: differences between actual distances from objects and RSS safe distances; 4D matrix
        :return: point-wise cost components: obstacles_costs, deviations_costs, jerk_costs.
        The tensor shape: N x M x 4, where N is trajectories number, M is trajectory length.
        """
        ''' OBSTACLES (Sigmoid cost from bounding-box) '''
        obstacles_costs = TrajectoryPlannerCosts.compute_obstacle_costs(ctrajectories, state, params,
                                                                        global_time_samples, predictor,
                                                                        reference_route)

        ''' DEVIATIONS FROM LANE/SHOULDER/ROAD '''
        deviations_costs = TrajectoryPlannerCosts.compute_deviation_costs(ftrajectories, params)

        ''' JERK COST '''
        jerk_costs = TrajectoryPlannerCosts.compute_jerk_costs(ctrajectories, params, dt)

        ''' safety (RSS) COST '''
        safety_costs = TrajectoryPlannerCosts.compute_safety_costs(safe_distances)

        return np.dstack((obstacles_costs, deviations_costs, jerk_costs, safety_costs))

    @staticmethod
    def compute_obstacle_costs(ctrajectories: CartesianExtendedTrajectories, state: State,
                               params: TrajectoryCostParams, global_time_samples: np.ndarray,
                               predictor: RoadFollowingPredictor, reference_route: FrenetSerret2DFrame):
        """
        :param ctrajectories: numpy tensor of trajectories in cartesian-frame
        :param state: the state object (that includes obstacles, etc.)
        :param params: parameters for the cost function (from behavioral layer)
        :param global_time_samples: [sec] time samples for prediction (global, not relative)
        :param predictor: predictor instance to use to compute future localizations for DyanmicObjects
        :param reference_route: reference route (GFF)
        :return: MxN matrix of obstacle costs per point, where N is trajectories number, M is trajectory length
        """
        # Filter close objects
        close_objects = [obs for obs in state.dynamic_objects
                   if np.linalg.norm([obs.x - state.ego_state.x, obs.y - state.ego_state.y]) < PLANNING_LOOKAHEAD_DIST]

        with prof.time_range('new_compute_obstacle_costs{objects: %d, ctraj_shape: %s}' % (len(close_objects), ctrajectories.shape)):
            if len(close_objects) == 0:
                return np.zeros((ctrajectories.shape[0], ctrajectories.shape[1]))

            # calculate objects' map_state
            # TODO: consider using map_state_on_host_lane
            objects_relative_fstates = np.array([obj.map_state.lane_fstate
                                                 for obj in close_objects if obj.cartesian_state is not None])

            # Predict objects' future movement, then project predicted objects' states to Cartesian frame
            # TODO: this assumes predictor works with frenet frames relative to ego-lane - figure out if this is how we want to do it in the future.
            objects_predicted_ftrajectories = predictor.predict_frenet_states(
                objects_relative_fstates, global_time_samples - state.ego_state.timestamp_in_sec)
            objects_predicted_ctrajectories = reference_route.ftrajectories_to_ctrajectories(objects_predicted_ftrajectories)

            objects_sizes = np.array([[obs.size.length, obs.size.width] for obs in close_objects])
            ego_size = np.array([state.ego_state.size.length, state.ego_state.size.width])

            # Compute the distance to the closest point in every object to ego's boundaries (on the length and width axes)
            distances = TrajectoryPlannerCosts.compute_distances_to_objects(
                ctrajectories, objects_predicted_ctrajectories, objects_sizes, ego_size)

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
        objects_H_inv_transposed_trimmed = objects_H_inv[..., :2, :]

        ego_points = ego_ctrajectories[:, :, [C_X, C_Y]]
        ego_points_ext = np.dstack((ego_points, np.ones(ego_points.shape[:2])))

        # Ego-center coordinates are projected onto the objects' reference frames [M, N, T, 2]
        # with M ego-trajectories, N objects, T timestamps.
        ego_centers_in_objs_frame = np.einsum('mti, ntji -> mntj', ego_points_ext, objects_H_inv_transposed_trimmed)

        # deduct ego and objects' half-sizes on both dimensions (to reflect objects' boundaries and not center-point)
        distances_from_ego_boundaries = np.abs(ego_centers_in_objs_frame) - 0.5 * (objects_sizes[:, np.newaxis] + ego_size)

        return distances_from_ego_boundaries

    @staticmethod
    def compute_safety_costs(safe_distances: np.array):
        """
        Given raw (not weighted) safety costs (based on RSS), sum it over obstacles and multiply by weight
        :param safe_distances: difference between actual distances and safe distances.
                                4D tensor: trajectories x objects x timestamps x lon/lat
        :return: MxN matrix of safety costs per point, where N is trajectories number, M is trajectory length.
        """
        # transfer lon & lat normalized safe distances to truncated sigmoid costs
        normalized_offset = safe_distances * np.array([LON_SAFETY_SIGMOID_K_PARAM, LAT_SAFETY_SIGMOID_K_PARAM])
        # the following sigmoid obtains values between 0 and 2.
        sigmoid_costs = np.divide(2., (1. + np.exp(np.minimum(normalized_offset, EXP_CLIP_TH))))
        # truncate the sigmoid by 1 and extract lon & lat costs, which are safe if 0 < cost < 1 and unsafe if cost = 1
        logit_costs = np.minimum(sigmoid_costs, 1)

        return SAFETY_SIGMOID_COST * logit_costs[..., 0] * logit_costs[..., 1]  # lon_costs * lat_costs

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
