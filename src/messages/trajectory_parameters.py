from enum import Enum

import numpy as np

from common_data.lcm.generatedFiles.gm_lcm import LcmNumpyArray
from common_data.lcm.generatedFiles.gm_lcm import LcmSigmoidFunctionParams
from common_data.lcm.generatedFiles.gm_lcm import LcmTrajectoryCostParams
from common_data.lcm.generatedFiles.gm_lcm import LcmTrajectoryParameters
from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from decision_making.src.messages.str_serializable import StrSerializable
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.types import C_V, Limits
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame


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

        lcm_msg.velocity_limits = LcmNumpyArray()
        lcm_msg.velocity_limits.num_dimensions = len(self.velocity_limits.shape)
        lcm_msg.velocity_limits.shape = list(self.velocity_limits.shape)
        lcm_msg.velocity_limits.length = self.velocity_limits.size
        lcm_msg.velocity_limits.data = self.velocity_limits.flat.__array__().tolist()

        lcm_msg.lon_acceleration_limits = LcmNumpyArray()
        lcm_msg.lon_acceleration_limits.num_dimensions = len(self.lon_acceleration_limits.shape)
        lcm_msg.lon_acceleration_limits.shape = list(self.lon_acceleration_limits.shape)
        lcm_msg.lon_acceleration_limits.length = self.lon_acceleration_limits.size
        lcm_msg.lon_acceleration_limits.data = self.lon_acceleration_limits.flat.__array__().tolist()

        lcm_msg.lat_acceleration_limits = LcmNumpyArray()
        lcm_msg.lat_acceleration_limits.num_dimensions = len(self.lat_acceleration_limits.shape)
        lcm_msg.lat_acceleration_limits.shape = list(self.lat_acceleration_limits.shape)
        lcm_msg.lat_acceleration_limits.length = self.lat_acceleration_limits.size
        lcm_msg.lat_acceleration_limits.data = self.lat_acceleration_limits.flat.__array__().tolist()

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
                 , np.ndarray(shape = tuple(lcmMsg.velocity_limits.shape)
                            , buffer = np.array(lcmMsg.velocity_limits.data)
                            , dtype = float)
                 , np.ndarray(shape = tuple(lcmMsg.lon_acceleration_limits.shape)
                            , buffer = np.array(lcmMsg.lon_acceleration_limits.data)
                            , dtype = float)
                 , np.ndarray(shape = tuple(lcmMsg.lat_acceleration_limits.shape)
                            , buffer = np.array(lcmMsg.lat_acceleration_limits.data)
                            , dtype = float))


class ReferenceRoute(PUBSUB_MSG_IMPL):
    """Members annotations for python 2 compliant classes"""
    points = np.ndarray
    T = np.ndarray
    N = np.ndarray
    k = np.ndarray
    k_tag = np.ndarray
    ds = float

    def __init__(self, points, T, N, k, k_tag, ds):
        # type: (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float) -> None
        """
        A data class that holds all information to build a FrenetSerret2DFrame
        for field descriptions, please refer to the FrenetSerret2DFrame class
        """
        self.points = points
        self.T = T
        self.N = N
        self.k = k
        self.k_tag = k_tag
        self.ds = ds

    @classmethod
    def from_frenet(cls, frenet):
        # type (FrenetSerret2DFrame) -> None
        return cls(frenet.O, frenet.T, frenet.N, frenet.k, frenet.k_tag, frenet.ds)

    def serialize(self):
        # type: () -> LcmReferenceRoute
        lcm_msg = LcmReferenceRoute()

        lcm_msg.points = LcmNumpyArray()
        lcm_msg.points.num_dimensions = len(self.points.shape)
        lcm_msg.points.shape = list(self.points.shape)
        lcm_msg.points.length = self.points.size
        lcm_msg.points.data = self.points.flat.__array__().tolist()

        lcm_msg.T = LcmNumpyArray()
        lcm_msg.T.num_dimensions = len(self.T.shape)
        lcm_msg.T.shape = list(self.T.shape)
        lcm_msg.T.length = self.T.size
        lcm_msg.T.data = self.T.flat.__array__().tolist()

        lcm_msg.N = LcmNumpyArray()
        lcm_msg.N.num_dimensions = len(self.N.shape)
        lcm_msg.N.shape = list(self.N.shape)
        lcm_msg.N.length = self.N.size
        lcm_msg.N.data = self.N.flat.__array__().tolist()

        lcm_msg.k = LcmNumpyArray()
        lcm_msg.k.num_dimensions = len(self.k.shape)
        lcm_msg.k.shape = list(self.k.shape)
        lcm_msg.k.length = self.k.size
        lcm_msg.k.data = self.k.flat.__array__().tolist()

        lcm_msg.k_tag = LcmNumpyArray()
        lcm_msg.k_tag.num_dimensions = len(self.k_tag.shape)
        lcm_msg.k_tag.shape = list(self.k_tag.shape)
        lcm_msg.k_tag.length = self.k_tag.size
        lcm_msg.k_tag.data = self.k_tag.flat.__array__().tolist()

        lcm_msg.ds = self.ds

        return lcm_msg

    @classmethod
    def deserialize(cls, lcmMsg):
        # type: (LcmReferenceRoute)->ReferenceRoute

        return cls(
            np.ndarray(shape=tuple(lcmMsg.points.shape)
                       , buffer=np.array(lcmMsg.points.data)
                       , dtype=float)
            , np.ndarray(shape=tuple(lcmMsg.T.shape)
                         , buffer=np.array(lcmMsg.T.data)
                         , dtype=float)
            , np.ndarray(shape=tuple(lcmMsg.N.shape)
                         , buffer=np.array(lcmMsg.N.data)
                         , dtype=float)
            , np.ndarray(shape=tuple(lcmMsg.k.shape)
                         , buffer=np.array(lcmMsg.k.data)
                         , dtype=float)
            , np.ndarray(shape=tuple(lcmMsg.k_tag.shape)
                         , buffer=np.array(lcmMsg.k_tag.data)
                         , dtype=float)
            , lcmMsg.ds)


class TrajectoryParams(PUBSUB_MSG_IMPL):
    """ Members annotations for python 2 compliant classes """
    strategy = TrajectoryPlanningStrategy
    reference_route = np.ndarray
    target_state = np.ndarray
    cost_params = TrajectoryCostParams
    time = float

    def __init__(self, strategy, reference_route, target_state, cost_params, time, bp_time):
        # type: (TrajectoryPlanningStrategy, ReferenceRoute, np.ndarray, TrajectoryCostParams, float)->None
        """
        The struct used for communicating the behavioral plan to the trajectory planner.
        :param reference_route: a reference route points (often the center of lane)
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

        lcm_msg.target_state = LcmNumpyArray()
        lcm_msg.target_state.num_dimensions = len(self.target_state.shape)
        lcm_msg.target_state.shape = list(self.target_state.shape)
        lcm_msg.target_state.length = self.target_state.size
        lcm_msg.target_state.data = self.target_state.flat.__array__().tolist()

        lcm_msg.cost_params = self.cost_params.serialize()

        lcm_msg.time = self.time
        lcm_msg.bp_time = self.bp_time

        return lcm_msg

    @classmethod
    def deserialize(cls, lcmMsg):
        # type: (LcmTrajectoryParameters)->TrajectoryParams
        return cls(TrajectoryPlanningStrategy(lcmMsg.strategy)
                 , ReferenceRoute.deserialize(lcmMsg.reference_route)
                 , np.ndarray(shape = tuple(lcmMsg.target_state.shape)
                            , buffer = np.array(lcmMsg.target_state.data)
                            , dtype = float)
                 , TrajectoryCostParams.deserialize(lcmMsg.cost_params)
                 , lcmMsg.time
                 , lcmMsg.bp_time)

