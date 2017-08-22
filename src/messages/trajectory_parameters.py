import numpy as np

from decision_making.src.messages.dds_typed_message import DDSTypedMsg
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy


class SigmoidFunctionParams(DDSTypedMsg):
    def __init__(self, w, k, offset):
        """
        A data class that corresponds to a parameterization of a sigmoid function
        :param w: considering sigmoid is: f(x) = w / (1 + exp(k * (x-offset)))
        :param k: considering sigmoid is: f(x) = w / (1 + exp(k * (x-offset)))
        :param offset: considering sigmoid is: f(x) = w / (1 + exp(k * (x-offset)))
        """
        self.w = w
        self.k = k
        self.offset = offset


class TrajectoryCostParams(DDSTypedMsg):
    def __init__(self, left_lane_cost: SigmoidFunctionParams, right_lane_cost: SigmoidFunctionParams,
                 left_road_cost: SigmoidFunctionParams, right_road_cost: SigmoidFunctionParams,
                 left_shoulder_cost: SigmoidFunctionParams, right_shoulder_cost: SigmoidFunctionParams,
                 obstacle_cost: SigmoidFunctionParams, dist_from_ref_sq_cost_coef: float,
                 velocity_limits: np.ndarray, acceleration_limits: np.ndarray):
        self.obstacle_cost = obstacle_cost
        self.left_lane_cost = left_lane_cost
        self.right_lane_cost = right_lane_cost
        self.left_shoulder_cost = left_shoulder_cost
        self.right_shoulder_cost = right_shoulder_cost
        self.left_road_cost = left_road_cost
        self.right_road_cost = right_road_cost
        self.dist_from_ref_sq_coef = dist_from_ref_sq_cost_coef
        self.velocity_limits = velocity_limits
        self.acceleration_limits = acceleration_limits


class TrajectoryParameters(DDSTypedMsg):
    # TODO: add <strategy> to IDL
    # TODO: add <time> to IDL
    def __init__(self, strategy: TrajectoryPlanningStrategy, reference_route: np.ndarray,
                 target_state: np.ndarray, cost_params: TrajectoryCostParams, time: float):
        """
        The struct used for communicating the behavioral plan to the trajectory planner.
        :param reference_route: of type np.ndarray, with rows of [(x ,y, theta)] where x, y, theta are floats
        :param target_state: of type np.array (x,y, theta, v) all of which are floats.
        :param cost_params: list of parameters for our predefined functions. TODO define this
        """
        self.reference_route = reference_route
        self.target_state = target_state
        self.cost_params = cost_params
        self.strategy = strategy
        self.time = time
