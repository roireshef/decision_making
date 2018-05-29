from abc import abstractmethod, ABCMeta
from typing import Optional

from decision_making.src.prediction.ego_aware_prediction.maneuver_spec import ManeuverSpec
from decision_making.src.state.state import State


class ManeuverClassifier(metaclass=ABCMeta):

    @abstractmethod
    def classify_maneuver(self, state: State, object_id: int, T_s: Optional[float] = None) -> ManeuverSpec:
        """
        Predicts the type of maneuver an object will execute
        :param state: world state
        :param object_id: of predicted object
        :return: maneuver specification of an object
        """
        pass
