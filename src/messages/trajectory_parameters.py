import numpy as np

from common_data.interface.Rte_Types.python.sub_structures import LcmTrajectoryParameters
from common_data.interface.Rte_Types.python.sub_structures.LcmSigmoidFunctionParams import \
    LcmSigmoidFunctionParams
from common_data.interface.Rte_Types.python.sub_structures.LcmTrajectoryCostParams import \
    LcmTrajectoryCostParams
from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.types import C_V, Limits
from decision_making.src.planning.types import FrenetState2D
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame


class SigmoidFunctionParams(PUBSUB_MSG_IMPL):
    """ Members annotations for python 2 compliant classes """
    w = float
    k = float
    offset = float

    def __init__(self, w, k, offset):
        # type: (float, float, float)->None
        """
        A data class that corresponds to a parametrization of a sigmoid function
        :param w: considering sigmoid is: f(x) = w / (1 + exp(k * (x-offset)))
        :param k: considering sigmoid is: f(x) = w / (1 + exp(k * (x-offset)))
        :param offset: considering sigmoid is: f(x) = w / (1 + exp(k * (x-offset)))
        """
        self.w = w
        self.k = k
        self.offset = offset

    def serialize(self):
        # type: () -> LcmSigmoidFunctionParams
        lcm_msg = LcmSigmoidFunctionParams()

        lcm_msg.w = self.w
        lcm_msg.k = self.k
        lcm_msg.offset = self.offset

        return lcm_msg

    @classmethod
    def deserialize(cls, lcmMsg):
        # type: (LcmSigmoidFunctionParams)->SigmoidFunctionParams
        return cls(lcmMsg.w, lcmMsg.k, lcmMsg.offset)


class TrajectoryCostParams(PUBSUB_MSG_IMPL):
    """ Members annotations for python 2 compliant classes """
    obstacle_cost_x = SigmoidFunctionParams
    obstacle_cost_y = SigmoidFunctionParams
    left_lane_cost = SigmoidFunctionParams
    right_lane_cost = SigmoidFunctionParams
    left_shoulder_cost = SigmoidFunctionParams
    right_shoulder_cost = SigmoidFunctionParams
    left_road_cost = SigmoidFunctionParams
    right_road_cost = SigmoidFunctionParams
    dist_from_goal_cost = SigmoidFunctionParams
    dist_from_goal_lat_factor = float
    lon_jerk_cost = float
    lat_jerk_cost = float
    velocity_limits = Limits
    lon_acceleration_limits = Limits
    lat_acceleration_limits = Limits

    def __init__(self, obstacle_cost_x, obstacle_cost_y, left_lane_cost, right_lane_cost, left_shoulder_cost,
                 right_shoulder_cost, left_road_cost, right_road_cost, dist_from_goal_cost, dist_from_goal_lat_factor,
                 lon_jerk_cost, lat_jerk_cost,
                 velocity_limits, lon_acceleration_limits, lat_acceleration_limits):
        # type:(SigmoidFunctionParams,SigmoidFunctionParams,SigmoidFunctionParams,SigmoidFunctionParams,SigmoidFunctionParams,SigmoidFunctionParams,SigmoidFunctionParams,SigmoidFunctionParams,SigmoidFunctionParams,float,float,float,Limits,Limits,Limits)->None
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
        :param obstacle_cost_x: defines the longitudinal sigmoid cost of obstacles
        :param obstacle_cost_y: defines the lateral sigmoid cost of obstacles
        :param left_lane_cost: defines the sigmoid cost of the left-side of the current lane
        :param right_lane_cost: defines the sigmoid cost of the right-side of the current lane
        :param left_shoulder_cost: defines the sigmoid cost of the left-shoulder of the road (physical boundary)
        :param right_shoulder_cost: defines the sigmoid cost of the right-shoulder of the road (physical boundary)
        :param left_road_cost: defines the sigmoid cost of the left-side of the road
        :param right_road_cost: defines the sigmoid cost of the right-side of the road
        :param dist_from_goal_cost: cost of distance from the target is a sigmoid.
        :param dist_from_goal_lat_factor: Weight of latitude vs. longitude in the dist from goal cost.
        :param lon_jerk_cost: longitudinal jerk cost
        :param lat_jerk_cost: lateral jerk cost
        :param velocity_limits: Limits of allowed velocity in [m/sec]
        :param lon_acceleration_limits: Limits of allowed longitudinal acceleration in [m/sec^2]
        :param lat_acceleration_limits: Limits of allowed signed lateral acceleration in [m/sec^2]
        """
        self.obstacle_cost_x = obstacle_cost_x
        self.obstacle_cost_y = obstacle_cost_y
        self.left_lane_cost = left_lane_cost
        self.right_lane_cost = right_lane_cost
        self.left_shoulder_cost = left_shoulder_cost
        self.right_shoulder_cost = right_shoulder_cost
        self.left_road_cost = left_road_cost
        self.right_road_cost = right_road_cost
        self.dist_from_goal_cost = dist_from_goal_cost
        self.dist_from_goal_lat_factor = dist_from_goal_lat_factor
        self.lon_jerk_cost = lon_jerk_cost
        self.lat_jerk_cost = lat_jerk_cost
        self.velocity_limits = velocity_limits
        self.lon_acceleration_limits = lon_acceleration_limits
        self.lat_acceleration_limits = lat_acceleration_limits

    def serialize(self):
        # type: ()-> LcmTrajectoryCostParams
        lcm_msg = LcmTrajectoryCostParams()

        lcm_msg.obstacle_cost_x = self.obstacle_cost_x.serialize()
        lcm_msg.obstacle_cost_y = self.obstacle_cost_y.serialize()
        lcm_msg.left_lane_cost = self.left_lane_cost.serialize()
        lcm_msg.right_lane_cost = self.right_lane_cost.serialize()
        lcm_msg.left_shoulder_cost = self.left_shoulder_cost.serialize()
        lcm_msg.right_shoulder_cost = self.right_shoulder_cost.serialize()
        lcm_msg.left_road_cost = self.left_road_cost.serialize()
        lcm_msg.right_road_cost = self.right_road_cost.serialize()
        lcm_msg.dist_from_goal_cost = self.dist_from_goal_cost.serialize()
        lcm_msg.dist_from_goal_lat_factor = self.dist_from_goal_lat_factor
        lcm_msg.lon_jerk_cost = self.lon_jerk_cost
        lcm_msg.lat_jerk_cost = self.lat_jerk_cost
        lcm_msg.velocity_limits = self.velocity_limits
        lcm_msg.lon_acceleration_limits = self.lon_acceleration_limits
        lcm_msg.lat_acceleration_limits = self.lat_acceleration_limits

        return lcm_msg

    @classmethod
    def deserialize(cls, lcmMsg):
        # type: (LcmTrajectoryCostParams)-> TrajectoryCostParams
        return cls(SigmoidFunctionParams.deserialize(lcmMsg.obstacle_cost_x)
                   , SigmoidFunctionParams.deserialize(lcmMsg.obstacle_cost_y)
                   , SigmoidFunctionParams.deserialize(lcmMsg.left_lane_cost)
                   , SigmoidFunctionParams.deserialize(lcmMsg.right_lane_cost)
                   , SigmoidFunctionParams.deserialize(lcmMsg.left_shoulder_cost)
                   , SigmoidFunctionParams.deserialize(lcmMsg.right_shoulder_cost)
                   , SigmoidFunctionParams.deserialize(lcmMsg.left_road_cost)
                   , SigmoidFunctionParams.deserialize(lcmMsg.right_road_cost)
                   , SigmoidFunctionParams.deserialize(lcmMsg.dist_from_goal_cost)
                   , lcmMsg.dist_from_goal_lat_factor
                   , lcmMsg.lon_jerk_cost
                   , lcmMsg.lat_jerk_cost
                   , lcmMsg.velocity_limits
                   , lcmMsg.lon_acceleration_limits
                   , lcmMsg.lat_acceleration_limits)


class TrajectoryParams(PUBSUB_MSG_IMPL):
    """ Members annotations for python 2 compliant classes """
    strategy = TrajectoryPlanningStrategy
    reference_route = GeneralizedFrenetSerretFrame
    target_state = np.ndarray
    cost_params = TrajectoryCostParams
    time = float

    def __init__(self, strategy, reference_route, target_state, cost_params, time, bp_time):
        # type: (TrajectoryPlanningStrategy, GeneralizedFrenetSerretFrame, FrenetState2D, TrajectoryCostParams, float, int)->None
        """
        The struct used for communicating the behavioral plan to the trajectory planner.
        :param reference_route: the frenet frame of the reference route (often the center of lane)
        :param target_state: the vector-representation of the target state to plan ego motion towards
        :param cost_params: list of parameters for the cost function of trajectory planner.
        :param strategy: trajectory planning strategy.
        :param time: trajectory planning time-frame
        """
        self.reference_route = reference_route
        self.target_state = target_state
        self.cost_params = cost_params
        self.strategy = strategy
        self.time = time
        self.bp_time = bp_time

    def __str__(self):
        return str(self.to_dict(left_out_fields=['reference_route']))

    @property
    def desired_velocity(self):
        return self.target_state[C_V]

    def serialize(self):
        # type: ()->LcmTrajectoryParameters
        lcm_msg = LcmTrajectoryParameters()

        lcm_msg.strategy = self.strategy.value

        lcm_msg.reference_route = self.reference_route.serialize()

        lcm_msg.target_state = self.target_state
        lcm_msg.cost_params = self.cost_params.serialize()

        lcm_msg.time = self.time
        lcm_msg.bp_time = self.bp_time

        return lcm_msg

    @classmethod
    def deserialize(cls, lcmMsg):
        # type: (LcmTrajectoryParameters)->TrajectoryParams
        return cls(TrajectoryPlanningStrategy(lcmMsg.strategy)
                   , GeneralizedFrenetSerretFrame.deserialize(lcmMsg.reference_route)
                   , lcmMsg.target_state
                   , TrajectoryCostParams.deserialize(lcmMsg.cost_params)
                   , lcmMsg.time
                   , lcmMsg.bp_time)
