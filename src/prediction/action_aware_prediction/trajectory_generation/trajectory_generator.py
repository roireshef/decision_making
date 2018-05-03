from abc import abstractmethod, ABCMeta

from decision_making.src.planning.types import FrenetTrajectory2D
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.prediction.action_aware_prediction.maneuver_spec import ManeuverSpec
from decision_making.src.state.state import DynamicObject


class TrajectoryGenerator(metaclass=ABCMeta):

    @abstractmethod
    def generate_trajectory(self, object_state: DynamicObject,
                            predicted_maneuver_spec: ManeuverSpec) -> \
            (FrenetSerret2DFrame, FrenetTrajectory2D):
        """
        Generate a trajectory in Frenet coordiantes, according to object's Frenet frame
        :param object_state: used for object localization and creation of the Frenet reference frame
        :param predicted_maneuver_spec: specification of the trajectory in Frenet frame.
        :return: Frenet reference frame, Trajectory in Frenet frame.
        """
        pass
