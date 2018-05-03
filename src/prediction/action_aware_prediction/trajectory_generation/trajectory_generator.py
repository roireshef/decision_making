from abc import abstractmethod, ABCMeta

from decision_making.src.planning.types import FrenetTrajectory2D
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.prediction.action_aware_prediction.maneuver_spec import ManeuverSpec
from decision_making.src.state.state import DynamicObject


class TrajectoryGenerator(metaclass=ABCMeta):

    @abstractmethod
    def generate_trajectory(self, object_state: DynamicObject, frenet_frame: FrenetSerret2DFrame,
                            predicted_maneuver_spec: ManeuverSpec) -> FrenetTrajectory2D:
        """
        Generate a trajectory in Frenet coordiantes, according to object's Frenet frame
        :param frenet_frame: Frenet reference frame of object
        :param object_state: used for object localization and creation of the Frenet reference frame
        :param predicted_maneuver_spec: specification of the trajectory in Frenet frame.
        :return: Trajectory in Frenet frame.
        """
        pass
