import numpy as np

from decision_making.src.messages.dds_typed_message import DDSTypedMsg
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy


class SigmoidFunctionParams(DDSTypedMsg):
    def __init__(self, w: float, k: float, offset: float):
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
        """
        This class holds all the parameters used to build the cost function of the trajectory planner.
        It is dynamically set and sent by the behavioral planner.
        :param left_lane_cost: defines the sigmoid cost of the left-side of the current lane
        :param right_lane_cost: defines the sigmoid cost of the right-side of the current lane
        :param left_road_cost: defines the sigmoid cost of the left-side of the road
        :param right_road_cost: defines the sigmoid cost of the right-side of the road
        :param left_shoulder_cost: defines the sigmoid cost of the left-shoulder of the road (physical boundary)
        :param right_shoulder_cost: defines the sigmoid cost of the right-shoulder of the road (physical boundary)
        :param obstacle_cost: defines the sigmoid cost of obstacles
        :param dist_from_ref_sq_cost_coef: if cost of distance from the reference route is C(x) = a*x^2, this is a.
        :param velocity_limits: 1D numpy array of [min allowed velocity, max allowed velocity]
        :param acceleration_limits: 1D numpy array of [min allowed acceleration, max allowed acceleration]
        """
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


class TrajectoryParams(DDSTypedMsg):
    def __init__(self, strategy: TrajectoryPlanningStrategy, reference_route: np.ndarray,
                 target_state: np.ndarray, cost_params: TrajectoryCostParams, time: float):
        """
        The struct used for communicating the behavioral plan to the trajectory planner.
        :param reference_route: of type np.ndarray, with rows of [(x ,y, theta)] where x, y, theta are floats
        :param target_state: of type np.array (x,y, theta, v) all of which are floats.
        :param cost_params: list of parameters for the cost function of trajectory planner.
        :param strategy: trajectory planning strategy.
        :param time: trajectory planning time-frame
        """
        self.reference_route = reference_route
        self.target_state = target_state
        self.cost_params = cost_params
        self.strategy = strategy
        self.time = time
