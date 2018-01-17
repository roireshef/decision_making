import numpy as np

from decision_making.src.messages.str_serializable import StrSerializable
from decision_making.src.planning.types import C_V, Limits
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy

from common_data.lcm.generatedFiles.gm_lcm import LcmTrajectoryParameters
from common_data.lcm.generatedFiles.gm_lcm import LcmSigmoidFunctionParams
from common_data.lcm.generatedFiles.gm_lcm import LcmTrajectoryCostParams
from common_data.lcm.generatedFiles.gm_lcm import LcmNumpyArray

class SigmoidFunctionParams(StrSerializable):
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

    def serialize(self) -> LcmSigmoidFunctionParams:
        lcm_msg = LcmSigmoidFunctionParams()

        lcm_msg.w = self.w
        lcm_msg.k = self.k
        lcm_msg.offset = self.offset

        return lcm_msg

    @classmethod
    def deserialize(cls, lcmMsg: LcmSigmoidFunctionParams):
        return cls(lcmMsg.w, lcmMsg.k, lcmMsg.offset)


class TrajectoryCostParams(StrSerializable):
    def __init__(self, left_lane_cost: SigmoidFunctionParams, right_lane_cost: SigmoidFunctionParams,
                 left_road_cost: SigmoidFunctionParams, right_road_cost: SigmoidFunctionParams,
                 left_shoulder_cost: SigmoidFunctionParams, right_shoulder_cost: SigmoidFunctionParams,
                 obstacle_cost_x: SigmoidFunctionParams,
                 obstacle_cost_y: SigmoidFunctionParams,
                 dist_from_goal_lon_sq_cost: float, dist_from_goal_lat_sq_cost: float,
                 dist_from_ref_sq_cost: float, velocity_limits: Limits, acceleration_limits: Limits):
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
        :param obstacle_cost_x: defines the longitudinal sigmoid cost of obstacles
        :param obstacle_cost_y: defines the lateral sigmoid cost of obstacles
        :param dist_from_goal_lon_sq_cost: cost of distance from the target longitude is C(x) = a*x^2, this is a.
        :param dist_from_goal_lat_sq_cost: cost of distance from the target latitude is C(x) = a*x^2, this is a.
        :param dist_from_ref_sq_cost: if cost of distance from the reference route is C(x) = a*x^2, this is a.
        :param velocity_limits: Limits of allowed velocity in [m/sec]
        :param acceleration_limits: Limits of allowed acceleration in [m/sec^2]
        """

        self.obstacle_cost_x = obstacle_cost_x
        self.obstacle_cost_y = obstacle_cost_y
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

    def serialize(self) -> LcmTrajectoryCostParams:
        lcm_msg = LcmTrajectoryCostParams()

        lcm_msg.obstacle_cost_x = self.obstacle_cost_x.serialize()
        lcm_msg.obstacle_cost_y = self.obstacle_cost_y.serialize()
        lcm_msg.left_lane_cost = self.left_lane_cost.serialize()
        lcm_msg.right_lane_cost = self.right_lane_cost.serialize()
        lcm_msg.left_shoulder_cost = self.left_shoulder_cost.serialize()
        lcm_msg.right_shoulder_cost = self.right_shoulder_cost.serialize()
        lcm_msg.left_road_cost = self.left_road_cost.serialize()
        lcm_msg.right_road_cost = self.right_road_cost.serialize()

        lcm_msg.dist_from_goal_lon_sq_cost = self.dist_from_goal_lon_sq_cost
        lcm_msg.dist_from_goal_lat_sq_cost = self.dist_from_goal_lat_sq_cost
        lcm_msg.dist_from_ref_sq_cost = self.dist_from_ref_sq_cost

        lcm_msg.velocity_limits = LcmNumpyArray()
        lcm_msg.velocity_limits.num_dimensions = len(self.velocity_limits.shape)
        lcm_msg.velocity_limits.shape = list(self.velocity_limits.shape)
        lcm_msg.velocity_limits.length = self.velocity_limits.size
        lcm_msg.velocity_limits.data = self.velocity_limits.flat.__array__().tolist()

        lcm_msg.acceleration_limits = LcmNumpyArray()
        lcm_msg.acceleration_limits.num_dimensions = len(self.acceleration_limits.shape)
        lcm_msg.acceleration_limits.shape = list(self.acceleration_limits.shape)
        lcm_msg.acceleration_limits.length = self.acceleration_limits.size
        lcm_msg.acceleration_limits.data = self.acceleration_limits.flat.__array__().tolist()

        return lcm_msg

    @classmethod
    def deserialize(cls, lcmMsg: LcmTrajectoryCostParams):
        return cls(SigmoidFunctionParams.deserialize(lcmMsg.obstacle_cost_x)
                 , SigmoidFunctionParams.deserialize(lcmMsg.obstacle_cost_y)
                 , SigmoidFunctionParams.deserialize(lcmMsg.left_lane_cost)
                 , SigmoidFunctionParams.deserialize(lcmMsg.right_lane_cost)
                 , SigmoidFunctionParams.deserialize(lcmMsg.left_shoulder_cost)
                 , SigmoidFunctionParams.deserialize(lcmMsg.right_shoulder_cost)
                 , SigmoidFunctionParams.deserialize(lcmMsg.left_road_cost)
                 , SigmoidFunctionParams.deserialize(lcmMsg.right_road_cost)
                 , lcmMsg.dist_from_goal_lon_sq_cost
                 , lcmMsg.dist_from_goal_lat_sq_cost
                 , lcmMsg.dist_from_ref_sq_cost
                 , np.ndarray(shape = tuple(lcmMsg.velocity_limits.shape)
                            , buffer = np.array(lcmMsg.velocity_limits.data)
                            , dtype = float)
                 , np.ndarray(shape = tuple(lcmMsg.acceleration_limits.shape)
                            , buffer = np.array(lcmMsg.acceleration_limits.data)
                            , dtype = float))

class TrajectoryParams(StrSerializable):
    def __init__(self, strategy: TrajectoryPlanningStrategy, reference_route: np.ndarray,
                 target_state: np.ndarray, cost_params: TrajectoryCostParams, time: float):
        """
        The struct used for communicating the behavioral plan to the trajectory planner.
        :param reference_route: a reference route (often the center of lane). A numpy array of the shape [-1, 2] where
        each row is a point (x, y) relative to the ego-coordinate-frame.
        :param target_state: A 1D numpy array of the desired ego-state to plan towards, represented in current
        ego-coordinate-frame
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
        return self.target_state[C_V]

    def serialize(self) -> LcmTrajectoryParameters:
        lcm_msg = LcmTrajectoryParameters()

        lcm_msg.strategy = self.strategy.value

        lcm_msg.reference_route = LcmNumpyArray()
        lcm_msg.reference_route.num_dimensions = len(self.reference_route.shape)
        lcm_msg.reference_route.shape = list(self.reference_route.shape)
        lcm_msg.reference_route.length = self.reference_route.size
        lcm_msg.reference_route.data = self.reference_route.flat.__array__().tolist()

        lcm_msg.target_state = LcmNumpyArray()
        lcm_msg.target_state.num_dimensions = len(self.target_state.shape)
        lcm_msg.target_state.shape = list(self.target_state.shape)
        lcm_msg.target_state.length = self.target_state.size
        lcm_msg.target_state.data = self.target_state.flat.__array__().tolist()

        lcm_msg.cost_params = self.cost_params.serialize()

        lcm_msg.time = self.time

        return lcm_msg

    @classmethod
    def deserialize(cls, lcmMsg: LcmTrajectoryParameters):
        return cls(TrajectoryPlanningStrategy(lcmMsg.strategy)
                 , np.ndarray(shape = tuple(lcmMsg.reference_route.shape)
                            , buffer = np.array(lcmMsg.reference_route.data)
                            , dtype = float)
                 , np.ndarray(shape = tuple(lcmMsg.target_state.shape)
                            , buffer = np.array(lcmMsg.target_state.data)
                            , dtype = float)
                 , TrajectoryCostParams.deserialize(lcmMsg.cost_params)
                 , lcmMsg.time)

