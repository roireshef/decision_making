import numpy as np

from common_data.lcm.generatedFiles.gm_lcm import LcmNumpyArray
from common_data.lcm.generatedFiles.gm_lcm import LcmTrajectoryData
from decision_making.src.global_constants import PUBSUB_MSG_IMPL


class TrajectoryPlanMsg(PUBSUB_MSG_IMPL):
    def __init__(self, timestamp, trajectory, current_speed):
        # type: (int, np.ndarray, float) -> ()
        """
        A discrete representation of the trajectory to follow - passed from TrajectoryPlanner to Controller
        :param timestamp: ego's timestamp on which the trajectory is based. Used for giving the controller a reference
        between the planned trajectory at current time, and the part to be executed due to time delays in the system.
        :param trajectory: numpy 2D array - 9 rows with each row containing <x, y, yaw, curvature, v> where x and y
         values are given in global coordinate frame, yaw and curvature are currently only placeholders, and v is the
         desired velocity of the vehicle on its actual longitudinal axis (the direction of its heading).
         Note: the first points corresponds to the current pose of the ego-vehicle at the time of plan, and only the
         following 8 points are points to follow further.
        :param current_speed: the current speed of the ego vehicle at the time of plan
        """
        self.timestamp = timestamp
        self.trajectory = trajectory
        self.current_speed = current_speed

    def serialize(self):
        # type: () -> LcmTrajectoryData
        lcm_msg = LcmTrajectoryData()

        # TODO: Uncomment when control message is ready
        #lcm_msg.timestamp = self.timestamp
        lcm_msg.trajectory = LcmNumpyArray()
        lcm_msg.trajectory.num_dimensions = len(self.trajectory.shape)
        lcm_msg.trajectory.shape = list(self.trajectory.shape)
        lcm_msg.trajectory.length = self.trajectory.size
        lcm_msg.trajectory.data = self.trajectory.flat.__array__().tolist()

        lcm_msg.current_speed = self.current_speed

        return lcm_msg

    @classmethod
    def deserialize(cls, lcmMsg):
        # type: (LcmTrajectoryData) -> TrajectoryPlanMsg

        return cls(0.0, # TODO: Currently a placeholder for the timestamp, until control will be ready for integration.
                        # TODO: Replace with lcmMsg.timestamp when control message is ready
                   np.ndarray(shape=tuple(lcmMsg.trajectory.shape)
                              , buffer=np.array(lcmMsg.trajectory.data)
                              , dtype=float)
                   , lcmMsg.current_speed)
