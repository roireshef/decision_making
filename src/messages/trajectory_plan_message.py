import numpy as np

from common_data.interface.py.idl_generated_files.dm.sub_structures.LcmNumpyArray import LcmNumpyArray
from common_data.interface.py.idl_generated_files.dm import LcmTrajectoryData
from common_data.interface.py.utils.serialization_utils import SerializationUtils
from decision_making.src.global_constants import PUBSUB_MSG_IMPL


class TrajectoryPlanMsg(PUBSUB_MSG_IMPL):
    ''' Members annotations for python 2 compliant classes '''
    timestamp = int
    trajectory = np.ndarray
    current_speed = float

    def __init__(self, timestamp, trajectory, current_speed):
        # type: (int, np.ndarray, float) -> None
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

        lcm_msg.timestamp = self.timestamp
        lcm_msg.trajectory = SerializationUtils.serialize_array(self.trajectory)
        lcm_msg.current_speed = self.current_speed

        return lcm_msg

    @classmethod
    def deserialize(cls, lcmMsg):
        # type: (LcmTrajectoryData) -> TrajectoryPlanMsg

        return cls(lcmMsg.timestamp,
                   SerializationUtils.deserialize_any_array(lcmMsg.trajectory),
                   lcmMsg.current_speed)
