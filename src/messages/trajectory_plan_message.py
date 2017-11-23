from decision_making.src.messages.dds_typed_message import DDSTypedMsg
import numpy as np


class TrajectoryPlanMsg(DDSTypedMsg):
    def __init__(self, trajectory: np.ndarray, current_speed: float):
        """
        A discrete representation of the trajectory to follow - passed from TrajectoryPlanner to Controller
        :param trajectory: numpy 2D array - 9 rows with each row containing <x, y, yaw, curvature, v> where x and y
         values are given in global coordinate frame, yaw and curvature are currently only placeholders, and v is the
         desired velocity of the vehicle on its actual longitudinal axis (the direction of its heading).
         Note: the first points corresponds to the current pose of the ego-vehicle at the time of plan, and only the
         following 8 points are points to follow further.
        :param current_speed: the current speed of the ego vehicle at the time of plan
        """
        self.trajectory = trajectory
        self.current_speed = current_speed
