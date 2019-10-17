from abc import abstractmethod, ABCMeta

from decision_making.src.prediction.ego_aware_prediction.maneuver_spec import ManeuverSpec
from decision_making.src.state.state import State


class ManeuverClassifier(metaclass=ABCMeta):

    @abstractmethod
    def classify_maneuver(self, state: State, object_id: int, maneuver_horizon: float) -> ManeuverSpec:
        """
        Predicts the type of maneuver an object will execute
        :param state: world state
        :param object_id: of the object to predict
        :param maneuver_horizon: the horizon of the maneuver to classify
        :return: maneuver specification of an object
        """
        pass
