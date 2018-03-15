import numpy as np

from decision_making.src.global_constants import TRAJECTORY_OBSTACLE_LOOKAHEAD, \
    EFFICIENCY_COST_DERIV_ZERO_DESIRED_RATIO, BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams
from decision_making.src.planning.performance_metrics.metric import Metric, PMState
from decision_making.src.planning.trajectory.cost_function import SigmoidDynamicBoxObstacle
from decision_making.src.planning.types import C_Y, FS_SV, FS_DX, CartesianExtendedTrajectories, C_A, C_K, C_V
from decision_making.src.planning.utils.math import Math



class SafetyMetric(Metric):
    @staticmethod
    def calc_pointwise_cost(pm_state: PMState, params: TrajectoryCostParams) -> np.ndarray:
        """
        calculate safety costs
        :param pm_state: contains the current state, trajectories, road parameters, predictor
        :param params: parameters for the cost function (from behavioral layer)
        :return: NxM matrix of obstacle costs per point, where N is trajectories number, M is trajectory length
        """
        offset = np.array([params.obstacle_cost_x.offset, params.obstacle_cost_y.offset])
        close_obstacles = \
            [SigmoidDynamicBoxObstacle.from_object(obj=obs, k=params.obstacle_cost_x.k, offset=offset,
                                                   time_samples=pm_state.time_samples, predictor=pm_state.predictor)
             for obs in pm_state.state.dynamic_objects
             if np.linalg.norm([obs.x - pm_state.state.ego_state.x, obs.y - pm_state.state.ego_state.y]) < TRAJECTORY_OBSTACLE_LOOKAHEAD]

        if len(close_obstacles) > 0:
            cost_per_obstacle = [obs.compute_cost_per_point(pm_state.ctrajectories[:, :, 0:(C_Y + 1)]) for obs in close_obstacles]
            obstacles_costs = np.sum(cost_per_obstacle, axis=0)
        else:
            obstacles_costs = np.zeros((pm_state.ctrajectories.shape[0], pm_state.ctrajectories.shape[1]))
        return obstacles_costs


class ComfortMetric(Metric):
    @staticmethod
    def calc_pointwise_cost(pm_state: PMState, params: TrajectoryCostParams) -> np.ndarray:
        """
        Compute point-wise jerk costs as weighted sum of longitudinal and lateral jerks
        :param pm_state: contains the current state, trajectories, road parameters, predictor
        :param params: parameters for the cost function (from behavioral layer)
        :return: NxM matrix of jerk costs per point, where N is trajectories number, M is trajectory length.
        """
        dt = pm_state.time_samples[1] - pm_state.time_samples[0]
        lon_jerks, lat_jerks = Jerk.compute_jerks(pm_state.ctrajectories, dt)
        lon_cost = params.lon_jerk_cost / (params.lon_jerk_cost + params.lat_jerk_cost)
        lat_cost = 1 - lon_cost
        jerk_costs = lon_cost * lon_jerks + lat_cost * lat_jerks
        return np.c_[np.zeros(jerk_costs.shape[0]), jerk_costs]


class EfficiencyMetric(Metric):
    @staticmethod
    def calc_pointwise_cost(pm_state: PMState, params: TrajectoryCostParams) -> np.ndarray:
        """
        calculate efficiency (velocity) cost by parabola function
        C(vel) = P(v) = a*v*v + b*v, where v = abs(1 - vel/vel_des), C(vel_des) = 0, C(0) = 1, C'(0)/C'(vel_des) = r
        :param pm_state: contains the current state, trajectories, road parameters, predictor
        :param params: parameters for the cost function (from behavioral layer)
        :return: NxM matrix of efficiency costs per point, where N is trajectories number, M is trajectory length.
        """
        r = EFFICIENCY_COST_DERIV_ZERO_DESIRED_RATIO  # C'(0)/C'(vel_des) = P'(1)/P'(0)
        # the following two lines are the solution of two equations on a and b: P(1) = 1, P'(1)/P'(0) = r
        a = (r-1)/(r+1)
        b = 2/(r+1)
        v = np.absolute(1 - pm_state.ftrajectories[:, :, FS_SV] / BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED)
        costs = v * (a * v + b)
        return costs


class LaneDeviationMetric(Metric):
    @staticmethod
    def calc_pointwise_cost(pm_state: PMState, params: TrajectoryCostParams) -> np.ndarray:
        """
        Compute point-wise deviation costs from the target lane (reference route).
        :param pm_state: contains the current state, trajectories, road parameters, predictor
        :param params: parameters for the cost function (from behavioral layer)
        :return: NxM matrix of deviation costs per point, where N is trajectories number, M is trajectory length.
        """
        # deviations to the left from the lane
        left_offsets = pm_state.ftrajectories[:, :, FS_DX] - params.left_lane_cost.offset
        left_deviations_costs = Math.clipped_sigmoid(left_offsets, 1, params.left_lane_cost.k)
        # deviations to the right from the lane
        right_offsets = np.negative(pm_state.ftrajectories[:, :, FS_DX]) - params.right_lane_cost.offset
        right_deviations_costs = Math.clipped_sigmoid(right_offsets, 1, params.right_lane_cost.k)
        return left_deviations_costs + right_deviations_costs


class ShoulderDeviationMetric(Metric):
    @staticmethod
    def calc_pointwise_cost(pm_state: PMState, params: TrajectoryCostParams) -> np.ndarray:
        """
        Compute point-wise deviation costs from the road shoulders (road margins).
        :param pm_state: contains the current state, trajectories, road parameters, predictor
        :param params: parameters for the cost function (from behavioral layer)
        :return: NxM matrix of deviation costs per point, where N is trajectories number, M is trajectory length.
        """
        # deviations to the left shoulder
        left_offsets = pm_state.ftrajectories[:, :, FS_DX] - params.left_shoulder_cost.offset
        left_deviations_costs = Math.clipped_sigmoid(left_offsets, 1, params.left_shoulder_cost.k)
        # deviations to the right shoulder
        right_offsets = np.negative(pm_state.ftrajectories[:, :, FS_DX]) - params.right_shoulder_cost.offset
        right_deviations_costs = Math.clipped_sigmoid(right_offsets, 1, params.right_shoulder_cost.k)
        return left_deviations_costs + right_deviations_costs


class RoadDeviationMetric(Metric):
    @staticmethod
    def calc_pointwise_cost(pm_state: PMState, params: TrajectoryCostParams) -> np.ndarray:
        """
        Compute point-wise deviation costs from the road (outside the shoulders).
        :param pm_state: contains the current state, trajectories, road parameters, predictor
        :param params: parameters for the cost function (from behavioral layer)
        :return: NxM matrix of deviation costs per point, where N is trajectories number, M is trajectory length.
        """
        # deviations to the left from the road
        left_offsets = pm_state.ftrajectories[:, :, FS_DX] - params.left_road_cost.offset
        left_deviations_costs = Math.clipped_sigmoid(left_offsets, 1, params.left_road_cost.k)
        # deviations to the right from the road
        right_offsets = np.negative(pm_state.ftrajectories[:, :, FS_DX]) - params.right_road_cost.offset
        right_deviations_costs = Math.clipped_sigmoid(right_offsets, 1, params.right_road_cost.k)
        return left_deviations_costs + right_deviations_costs


class RightLaneMetric(Metric):
    @staticmethod
    def calc_pointwise_cost(pm_state: PMState, params: TrajectoryCostParams) -> np.ndarray:
        """
        calculate cost for using non-right lane
        :param pm_state: contains the current state, trajectories, road parameters, predictor
        :param params: parameters for the cost function (from behavioral layer)
        :return: NxM matrix of deviation costs per point, where N is trajectories number, M is trajectory length.
        """
        float_lane_num = (pm_state.ftrajectories[:, :, FS_DX] + pm_state.reference_route_lat) / pm_state.lane_width
        return np.floor(float_lane_num).astype(int)


class GoalAchievementMetric(Metric):
    @staticmethod
    def calc_pointwise_cost(pm_state: PMState, params: TrajectoryCostParams) -> np.ndarray:
        # TODO: add more goal parameters to PMState: goal_road_id, goal_lanes_set, <goal_longitude> (optional)
        """
        Calculate cost for goal missing. Any trajectory that misses the goal has cost 1, otherwise 0.
        :param pm_state: contains the current state, trajectories, road parameters, predictor
        :param params: parameters for the cost function (from behavioral layer)
        :return: Nx1 matrix of missing goal costs, where N is trajectories number.
        """
        pass


class Jerk:
    @staticmethod
    def compute_jerks(ctrajectories: CartesianExtendedTrajectories, dt: float) -> [np.array, np.array]:
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
