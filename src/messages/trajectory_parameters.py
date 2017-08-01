import numpy as np

from src.messages.dds_message import DDSTypedMsg


class TrajectoryParamsMsg(DDSTypedMsg):
    def __init__(self, reference_route: np.ndarray, target_state: np.array, cost_params: TrajectoryCostParams):
        """
        The struct used for communicating the behavioral plan to the trajectory planner.
        :param reference_route: of type np.ndarray, with rows of [(x ,y, theta)] where x, y, theta are floats
        :param target_state: of type np.array (x,y, theta, v) all of which are floats.
        :param cost_params: list of parameters for our predefined functions. TODO define this
        """
        self.reference_route = reference_route
        self.target_state = target_state
        self.cost_params = cost_params


class TrajectoryCostParams(DDSTypedMsg):
    def __init__(self, time: float, ref_deviation_weight: float, lane_deviation_weight: float, obstacle_weight: float,
                 left_lane_offset: float, right_lane_offset: float, left_deviation_exp: float,
                 right_deviation_exp: float, obstacle_offset: float, obstacle_exp: float, v_x_min_limit: float,
                 v_x_max_limit: float, a_x_min_limit: float, a_x_max_limit: float):
        self.time = time
        self.ref_deviation_weight = ref_deviation_weight
        self.lane_deviation_weight = lane_deviation_weight
        self.obstacle_weight = obstacle_weight
        self.left_lane_offset = left_lane_offset
        self.right_lane_offset = right_lane_offset
        self.left_deviation_exp = left_deviation_exp
        self.right_deviation_exp = right_deviation_exp
        self.obstacle_offset = obstacle_offset
        self.obstacle_exp = obstacle_exp
        self.v_x_min_limit = v_x_min_limit
        self.v_x_max_limit = v_x_max_limit
        self.a_x_min_limit = a_x_min_limit
        self.a_x_max_limit = a_x_max_limit
