import numpy as np
from common_data.interface.Rte_Types.python.sub_structures.TsSYS_SigmoidFunctionParams import \
    TsSYSSigmoidFunctionParams
from common_data.interface.Rte_Types.python.sub_structures.TsSYS_TrajectoryCostParams import \
    TsSYSTrajectoryCostParams
from common_data.interface.Rte_Types.python.sub_structures.TsSYS_TrajectoryParameters import TsSYSTrajectoryParameters
from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.types import C_V, Limits
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame


class SigmoidFunctionParams(PUBSUB_MSG_IMPL):
    """ Members annotations for python 2 compliant classes """
    w = float
    k = float
    offset = float

    def __init__(self, w: float, k: float, offset: float):
        """
        A data class that corresponds to a parametrization of a sigmoid function
        :param w: considering sigmoid is: f(x) = w / (1 + exp(k * (x-offset)))
        :param k: considering sigmoid is: f(x) = w / (1 + exp(k * (x-offset)))
        :param offset: considering sigmoid is: f(x) = w / (1 + exp(k * (x-offset)))
        """
        self.w = w
        self.k = k
        self.offset = offset

    def serialize(self) -> TsSYSSigmoidFunctionParams:
        pubsub_msg = TsSYSSigmoidFunctionParams()

        pubsub_msg.e_Prm_SigmoidW = self.w
        pubsub_msg.e_Prm_SigmoidK = self.k
        pubsub_msg.e_Prm_SigmoidOffset = self.offset

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSSigmoidFunctionParams)->SigmoidFunctionParams
        return cls(pubsubMsg.e_Prm_SigmoidW, pubsubMsg.e_Prm_SigmoidK, pubsubMsg.e_Prm_SigmoidOffset)


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
    dist_from_goal_cost = float
    deviation_from_target_time_cost = float
    lon_jerk_cost_weight = float
    lat_jerk_cost_weight = float
    velocity_limits = Limits
    lon_acceleration_limits = Limits
    lat_acceleration_limits = Limits
    desired_velocity = float
    def __init__(self, obstacle_cost_x, obstacle_cost_y, left_lane_cost, right_lane_cost, left_shoulder_cost,
                 right_shoulder_cost, left_road_cost, right_road_cost, dist_from_goal_cost, deviation_from_target_time_cost,
                 lon_jerk_cost_weight, lat_jerk_cost_weight,
                 velocity_limits, lon_acceleration_limits, lat_acceleration_limits, desired_velocity):
        # type:(SigmoidFunctionParams,SigmoidFunctionParams,SigmoidFunctionParams,SigmoidFunctionParams,SigmoidFunctionParams,SigmoidFunctionParams,SigmoidFunctionParams,SigmoidFunctionParams,SigmoidFunctionParams,float,float,float,Limits,Limits,Limits, int)->None
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
        :param dist_from_goal_cost: weight of squared distance cost from the target
        :param deviation_from_target_time_cost: Weight of deviation from the target horizon time.
        :param lon_jerk_cost_weight: longitudinal jerk cost
        :param lat_jerk_cost_weight: lateral jerk cost
        :param velocity_limits: Limits of allowed velocity in [m/sec]
        :param lon_acceleration_limits: Limits of allowed longitudinal acceleration in [m/sec^2]
        :param lat_acceleration_limits: Limits of allowed signed lateral acceleration in [m/sec^2]
        :param desired_velocity : The longitudinal velocity the vehicle should aim for [m/sec]
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
        self.deviation_from_target_time_cost = deviation_from_target_time_cost
        self.lon_jerk_cost_weight = lon_jerk_cost_weight
        self.lat_jerk_cost_weight = lat_jerk_cost_weight
        self.velocity_limits = velocity_limits
        self.lon_acceleration_limits = lon_acceleration_limits
        self.lat_acceleration_limits = lat_acceleration_limits
        self.desired_velocity = desired_velocity

    def serialize(self):
        # type: ()-> TsSYSTrajectoryCostParams
        pubsub_msg = TsSYSTrajectoryCostParams()

        pubsub_msg.s_ObstacleCostX = self.obstacle_cost_x.serialize()
        pubsub_msg.s_ObstacleCostY = self.obstacle_cost_y.serialize()
        pubsub_msg.s_LeftLaneCost = self.left_lane_cost.serialize()
        pubsub_msg.s_RightLaneCost = self.right_lane_cost.serialize()
        pubsub_msg.s_LeftShoulderCost = self.left_shoulder_cost.serialize()
        pubsub_msg.s_RightShoulderCost = self.right_shoulder_cost.serialize()
        pubsub_msg.s_LeftRoadCost = self.left_road_cost.serialize()
        pubsub_msg.s_RightRoadCost = self.right_road_cost.serialize()
        pubsub_msg.s_DistanceFromGoalCost = self.dist_from_goal_cost
        pubsub_msg.e_l_DeviationFromTargetTimeCost = self.deviation_from_target_time_cost
        pubsub_msg.e_Wt_LonJerkCostWeight = self.lon_jerk_cost_weight
        pubsub_msg.e_Wt_LatJerkCostWeight = self.lat_jerk_cost_weight
        pubsub_msg.e_v_VelocityLimits = self.velocity_limits
        pubsub_msg.e_a_LonAccelerationLimits = self.lon_acceleration_limits
        pubsub_msg.e_a_LatAccelerationLimits = self.lat_acceleration_limits
        pubsub_msg.e_v_DesiredVelocity = self.desired_velocity

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSTrajectoryCostParams)-> TrajectoryCostParams
        return cls(SigmoidFunctionParams.deserialize(pubsubMsg.s_ObstacleCostX)
                   , SigmoidFunctionParams.deserialize(pubsubMsg.s_ObstacleCostY)
                   , SigmoidFunctionParams.deserialize(pubsubMsg.s_LeftLaneCost)
                   , SigmoidFunctionParams.deserialize(pubsubMsg.s_RightLaneCost)
                   , SigmoidFunctionParams.deserialize(pubsubMsg.s_LeftShoulderCost)
                   , SigmoidFunctionParams.deserialize(pubsubMsg.s_RightShoulderCost)
                   , SigmoidFunctionParams.deserialize(pubsubMsg.s_LeftRoadCost)
                   , SigmoidFunctionParams.deserialize(pubsubMsg.s_RightRoadCost)
                   , pubsubMsg.s_DistanceFromGoalCost
                   , pubsubMsg.e_l_DeviationFromTargetTimeCost
                   , pubsubMsg.e_Wt_LonJerkCostWeight
                   , pubsubMsg.e_Wt_LatJerkCostWeight
                   , pubsubMsg.e_v_VelocityLimits
                   , pubsubMsg.e_a_LonAccelerationLimits
                   , pubsubMsg.e_a_LatAccelerationLimits
                   , pubsubMsg.e_v_DesiredVelocity)


class TrajectoryParams(PUBSUB_MSG_IMPL):
    """ Members annotations for python 2 compliant classes """
    strategy = TrajectoryPlanningStrategy
    reference_route = GeneralizedFrenetSerretFrame
    target_state = np.ndarray
    cost_params = TrajectoryCostParams
    target_time = float
    bp_time = int

    def __init__(self, strategy, reference_route, target_state, cost_params, target_time, trajectory_end_time, bp_time):
        # type: (TrajectoryPlanningStrategy, GeneralizedFrenetSerretFrame, np.ndarray, TrajectoryCostParams, float, float, int)->None
        """
        The struct used for communicating the behavioral plan to the trajectory planner.
        :param reference_route: the frenet frame of the reference route (often the center of lane)
        :param target_state: the vector-representation of the target state to plan ego motion towards
        :param cost_params: list of parameters for the cost function of trajectory planner.
        :param strategy: trajectory planning strategy.
        :param target_time: the absolute timestamp in [sec] that represent the time of arrival to the terminal boundary
        condition (TP optimization problem will end in this time)
        :param trajectory_end_time: the timestamp in [sec] in which the actual trajectory will end (including padding
        done in TP, which is a consequence of Control's requirement to have a long enough trajectory, i.e. according to
        MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON)
        :param bp_time: absolute time of the state that BP planned on.
        """
        self.reference_route = reference_route
        self.target_state = target_state
        self.cost_params = cost_params
        self.strategy = strategy
        self.target_time = target_time
        self.trajectory_end_time = trajectory_end_time
        self.bp_time = bp_time

    def __str__(self):
        return str(self.to_dict(left_out_fields=['reference_route']))

    @property
    def desired_velocity(self):
        return self.target_state[C_V]

    def serialize(self):
        # type: ()->TsSYSTrajectoryParameters
        pubsub_msg = TsSYSTrajectoryParameters()

        pubsub_msg.e_e_Strategy = self.strategy.value

        pubsub_msg.s_ReferenceRoute = self.reference_route.serialize()

        pubsub_msg.a_TargetState = self.target_state
        pubsub_msg.s_CostParams = self.cost_params.serialize()

        pubsub_msg.e_t_TargetTime = self.target_time
        pubsub_msg.e_t_TrajectoryEndTime = self.trajectory_end_time
        pubsub_msg.e_Cnt_BPTime = self.bp_time

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSTrajectoryParameters)->TrajectoryParams
        return cls(TrajectoryPlanningStrategy(pubsubMsg.e_e_Strategy)
                   , GeneralizedFrenetSerretFrame.deserialize(pubsubMsg.s_ReferenceRoute)
                   , pubsubMsg.a_TargetState
                   , TrajectoryCostParams.deserialize(pubsubMsg.s_CostParams)
                   , pubsubMsg.e_t_TargetTime
                   , pubsubMsg.e_t_TrajectoryEndTime
                   , pubsubMsg.e_Cnt_BPTime)
