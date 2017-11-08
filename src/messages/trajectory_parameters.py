import numpy as np

from decision_making.src.messages.dds_typed_message import DDSTypedMsg
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.utils.columns import EGO_V


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
                 obstacle_cost: SigmoidFunctionParams,
                 dist_from_goal_lon_sq_cost: float, dist_from_goal_lat_sq_cost: float,
                 dist_from_ref_sq_cost: float, velocity_limits: np.ndarray, acceleration_limits: np.ndarray):
        """
        This class holds all the parameters used to build the cost function of the trajectory planner.
        It is dynamically set and sent by the behavioral planner.
        Important: there is an AMBIGUITY in the offset parameter of the SIGMOID function.
            - In the lane/shoulder/road case: the offset parameter defines the (fixed) lateral distance
                in [m] between the reference trajectory and the lane/shoulder/road
            - In the objects case: the offset parameter defines a (fixed) dilation in [m] of an object
                (i.e. the offset of the sigmoid border from the objects's bounding box, both in the
                length and width).
                This can be used to keep a certain margin from any object, specifically useful when
                treating the ego vehicle as a point in space, and dilating the other objects by it's width.
        :param left_lane_cost: defines the sigmoid cost of the left-side of the current lane
        :param right_lane_cost: defines the sigmoid cost of the right-side of the current lane
        :param left_road_cost: defines the sigmoid cost of the left-side of the road
        :param right_road_cost: defines the sigmoid cost of the right-side of the road
        :param left_shoulder_cost: defines the sigmoid cost of the left-shoulder of the road (physical boundary)
        :param right_shoulder_cost: defines the sigmoid cost of the right-shoulder of the road (physical boundary)
        :param obstacle_cost: defines the sigmoid cost of obstacles
        :param dist_from_goal_lon_sq_cost: cost of distance from the target longitude is C(x) = a*x^2, this is a.
        :param dist_from_goal_lat_sq_cost: cost of distance from the target latitude is C(x) = a*x^2, this is a.
        :param dist_from_ref_sq_cost: if cost of distance from the reference route is C(x) = a*x^2, this is a.
        :param velocity_limits: 1D numpy array of [min allowed velocity, max allowed velocity] in [m/sec]
        :param acceleration_limits: 1D numpy array of [min allowed acceleration, max allowed acceleration] in [m/sec^2]
        """
        self.obstacle_cost = obstacle_cost
        self.left_lane_cost = left_lane_cost
        self.right_lane_cost = right_lane_cost
        self.left_shoulder_cost = left_shoulder_cost
        self.right_shoulder_cost = right_shoulder_cost
        self.left_road_cost = left_road_cost
        self.right_road_cost = right_road_cost
        self.dist_from_goal_lon_sq_cost = dist_from_goal_lon_sq_cost
        self.dist_from_goal_lat_sq_cost = dist_from_goal_lat_sq_cost
        self.dist_from_ref_sq_cost = dist_from_ref_sq_cost
        self.velocity_limits = velocity_limits
        self.acceleration_limits = acceleration_limits


class TrajectoryParams(DDSTypedMsg):
    def __init__(self, strategy: TrajectoryPlanningStrategy, reference_route: np.ndarray,
                 target_state: np.ndarray, cost_params: TrajectoryCostParams, time: float):
        """
        The struct used for communicating the behavioral plan to the trajectory planner.
        :param reference_route: a reference route (often the center of lane). A numpy array of the shape [-1, 2] where
        each row is a point (x, y) relative to the ego-coordinate-frame.
        :param target_state: A 1D numpy array of the desired ego-state to plan towards, represented in current
        ego-coordinate-frame (see EGO_* in planning.utils.columns.py for the fields)
        :param cost_params: list of parameters for the cost function of trajectory planner.
        :param strategy: trajectory planning strategy.
        :param time: trajectory planning time-frame
        """
        self.reference_route = reference_route
        self.target_state = target_state
        self.cost_params = cost_params
        self.strategy = strategy
        self.time = time

    @property
    def desired_velocity(self):
        return self.target_state[EGO_V]