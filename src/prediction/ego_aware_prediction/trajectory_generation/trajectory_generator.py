from abc import abstractmethod, ABCMeta

from decision_making.src.planning.trajectory.samplable_werling_trajectory import SamplableWerlingTrajectory
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.prediction.ego_aware_prediction.maneuver_spec import ManeuverSpec


class TrajectoryGenerator(metaclass=ABCMeta):

    @abstractmethod
    def generate_trajectory(self, timestamp_in_sec: float, frenet_frame: FrenetSerret2DFrame,
                            predicted_maneuver_spec: ManeuverSpec) -> SamplableWerlingTrajectory:
        """
        Generate a trajectory in Frenet coordinates, according to object's Frenet frame
        :param frenet_frame: Frenet reference frame of object
        :param timestamp_in_sec: [sec] global timestamp *in seconds* to use as a reference
                (other timestamps will be given relative to it)
        :param predicted_maneuver_spec: specification of the trajectory in Frenet frame.
        :return: Trajectory in Frenet frame.
        """
        pass
