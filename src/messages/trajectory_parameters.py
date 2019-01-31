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
    e_Prm_SigmoidW = float
    e_Prm_SigmoidK = float
    e_Prm_SigmoidOffset = float

    def __init__(self, w: float, k: float, offset: float):
        """
        A data class that corresponds to a parametrization of a sigmoid function
        :param w: considering sigmoid is: f(x) = w / (1 + exp(k * (x-offset)))
        :param k: considering sigmoid is: f(x) = w / (1 + exp(k * (x-offset)))
        :param offset: considering sigmoid is: f(x) = w / (1 + exp(k * (x-offset)))
        """
        self.e_Prm_SigmoidW = w
        self.e_Prm_SigmoidK = k
        self.e_Prm_SigmoidOffset = offset

    def serialize(self) -> TsSYSSigmoidFunctionParams:
        pubsub_msg = TsSYSSigmoidFunctionParams()

        pubsub_msg.e_Prm_SigmoidW = self.e_Prm_SigmoidW
        pubsub_msg.e_Prm_SigmoidK = self.e_Prm_SigmoidK
        pubsub_msg.e_Prm_SigmoidOffset = self.e_Prm_SigmoidOffset

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSSigmoidFunctionParams)->SigmoidFunctionParams
        return cls(pubsubMsg.e_Prm_SigmoidW, pubsubMsg.e_Prm_SigmoidK, pubsubMsg.e_Prm_SigmoidOffset)


class TrajectoryCostParams(PUBSUB_MSG_IMPL):
    """ Members annotations for python 2 compliant classes """
    s_ObstacleCostX = SigmoidFunctionParams
    s_ObstacleCostY = SigmoidFunctionParams
    s_LeftLaneCost = SigmoidFunctionParams
    s_RightLaneCost = SigmoidFunctionParams
    s_LeftShoulderCost = SigmoidFunctionParams
    s_RightShoulderCost = SigmoidFunctionParams
    s_LeftRoadCost = SigmoidFunctionParams
    s_RightRoadCost = SigmoidFunctionParams
    s_DistanceFromGoalCost = SigmoidFunctionParams
    e_l_DistFromGoalLatFactor = float
    e_Wt_LonJerkCostWeight = float
    e_Wt_LatJerkCostWeight = float
    e_v_VelocityLimits = Limits
    e_a_LonAccelerationLimits = Limits
    e_a_LatAccelerationLimits = Limits

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
        self.s_ObstacleCostX = obstacle_cost_x
        self.s_ObstacleCostY = obstacle_cost_y
        self.s_LeftLaneCost = left_lane_cost
        self.s_RightLaneCost = right_lane_cost
        self.s_LeftShoulderCost = left_shoulder_cost
        self.s_RightShoulderCost = right_shoulder_cost
        self.s_LeftRoadCost = left_road_cost
        self.s_RightRoadCost = right_road_cost
        self.s_DistanceFromGoalCost = dist_from_goal_cost
        self.e_l_DistFromGoalLatFactor = dist_from_goal_lat_factor
        self.e_Wt_LonJerkCostWeight = lon_jerk_cost
        self.e_Wt_LatJerkCostWeight = lat_jerk_cost
        self.e_v_VelocityLimits = velocity_limits
        self.e_a_LonAccelerationLimits = lon_acceleration_limits
        self.e_a_LatAccelerationLimits = lat_acceleration_limits

    def serialize(self):
        # type: ()-> TsSYSTrajectoryCostParams
        pubsub_msg = TsSYSTrajectoryCostParams()

        pubsub_msg.s_ObstacleCostX = self.s_ObstacleCostX.serialize()
        pubsub_msg.s_ObstacleCostY = self.s_ObstacleCostY.serialize()
        pubsub_msg.s_LeftLaneCost = self.s_LeftLaneCost.serialize()
        pubsub_msg.s_RightLaneCost = self.s_RightLaneCost.serialize()
        pubsub_msg.s_LeftShoulderCost = self.s_LeftShoulderCost.serialize()
        pubsub_msg.s_RightShoulderCost = self.s_RightShoulderCost.serialize()
        pubsub_msg.s_LeftRoadCost = self.s_LeftRoadCost.serialize()
        pubsub_msg.s_RightRoadCost = self.s_RightRoadCost.serialize()
        pubsub_msg.s_DistanceFromGoalCost = self.s_DistanceFromGoalCost.serialize()
        pubsub_msg.e_l_DistFromGoalLatFactor = self.e_l_DistFromGoalLatFactor
        pubsub_msg.e_Wt_LonJerkCostWeight = self.e_Wt_LonJerkCostWeight
        pubsub_msg.e_Wt_LatJerkCostWeight = self.e_Wt_LatJerkCostWeight
        pubsub_msg.e_v_VelocityLimits = self.e_v_VelocityLimits
        pubsub_msg.e_a_LonAccelerationLimits = self.e_a_LonAccelerationLimits
        pubsub_msg.e_a_LatAccelerationLimits = self.e_a_LatAccelerationLimits

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
                   , SigmoidFunctionParams.deserialize(pubsubMsg.s_DistanceFromGoalCost)
                   , pubsubMsg.e_l_DistFromGoalLatFactor
                   , pubsubMsg.e_Wt_LonJerkCostWeight
                   , pubsubMsg.e_Wt_LatJerkCostWeight
                   , pubsubMsg.e_v_VelocityLimits
                   , pubsubMsg.e_a_LonAccelerationLimits
                   , pubsubMsg.e_a_LatAccelerationLimits)


class TrajectoryParams(PUBSUB_MSG_IMPL):
    """ Members annotations for python 2 compliant classes """
    e_e_Strategy = TrajectoryPlanningStrategy
    s_ReferenceRoute = GeneralizedFrenetSerretFrame
    a_TargetState = np.ndarray
    s_CostParams = TrajectoryCostParams
    e_t_Time = float

    def __init__(self, strategy, reference_route, target_state, cost_params, time, bp_time):
        # type: (TrajectoryPlanningStrategy, GeneralizedFrenetSerretFrame, np.ndarray, TrajectoryCostParams, float)->None
        """
        The struct used for communicating the behavioral plan to the trajectory planner.
        :param reference_route: the frenet frame of the reference route (often the center of lane)
        :param target_state: the vector-representation of the target state to plan ego motion towards
        :param cost_params: list of parameters for the cost function of trajectory planner.
        :param strategy: trajectory planning strategy.
        :param time: trajectory planning time-frame
        """
        self.s_ReferenceRoute = reference_route
        self.a_TargetState = target_state
        self.s_CostParams = cost_params
        self.e_e_Strategy = strategy
        self.e_t_Time = time
        self.e_Cnt_BPTime = bp_time

    def __str__(self):
        return str(self.to_dict(left_out_fields=['reference_route']))

    @property
    def desired_velocity(self):
        return self.a_TargetState[C_V]

    def serialize(self):
        # type: ()->TsSYSTrajectoryParameters
        pubsub_msg = TsSYSTrajectoryParameters()

        pubsub_msg.e_e_Strategy = self.e_e_Strategy.value

        pubsub_msg.s_ReferenceRoute = self.s_ReferenceRoute.serialize()

        pubsub_msg.a_TargetState = self.a_TargetState
        pubsub_msg.s_CostParams = self.s_CostParams.serialize()

        pubsub_msg.e_t_Time = self.e_t_Time
        pubsub_msg.e_Cnt_BPTime = self.e_Cnt_BPTime

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSTrajectoryParameters)->TrajectoryParams
        return cls(TrajectoryPlanningStrategy(pubsubMsg.e_e_Strategy)
                   , GeneralizedFrenetSerretFrame.deserialize(pubsubMsg.s_ReferenceRoute)
                   , pubsubMsg.a_TargetState
                   , TrajectoryCostParams.deserialize(pubsubMsg.s_CostParams)
                   , pubsubMsg.e_t_Time
                   , pubsubMsg.e_Cnt_BPTime)
