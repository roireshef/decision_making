import numpy as np

from common_data.lcm.generatedFiles.gm_lcm import LcmTrajectoryData
from common_data.lcm.generatedFiles.gm_lcm import LcmNumpyArray

class TrajectoryPlanMsg:
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

    def serialize(self) -> LcmTrajectoryData:
        lcm_msg = LcmTrajectoryData()

        lcm_msg.trajectory = LcmNumpyArray()
        lcm_msg.trajectory.num_dimensions = len(self.trajectory.shape)
        lcm_msg.trajectory.shape = list(self.trajectory.shape)
        lcm_msg.trajectory.length = self.trajectory.size
        lcm_msg.trajectory.data = self.trajectory.flat.__array__().tolist()

        lcm_msg.current_speed = self.current_speed

        return lcm_msg

    @classmethod
    def deserialize(cls, lcmMsg: LcmTrajectoryData):
        return cls(np.ndarray(shape = tuple(lcmMsg.trajectory.shape)
                            , buffer = np.array(lcmMsg.trajectory.data)
                            , dtype = float)
                 , lcmMsg.current_speed)

