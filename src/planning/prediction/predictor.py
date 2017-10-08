import numpy as np

from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.state.state import DynamicObject
from mapping.src.model.map_api import MapAPI


class Predictor:
    """
    Base class for predictors of dynamic objects
    """

    @staticmethod
    def predict(dynamic_object: DynamicObject,  predicted_timestamps: np.ndarray, map_api: MapAPI,
                nav_plan:NavigationPlanMsg) -> np.ndarray:
        """
        :param dynamic_object: in map coordinates
        :param predicted_timestamps: np array of timestamps to predict for. In ascending order.
        :param map_api: used in order to get the predicted trajectory and center lanes in map  coordinates
        :param nav_plan: predicted navigation plan of the object
        :return: predicted object's locations in global map coordinates np.array([x, y, theta, vel])
        """

