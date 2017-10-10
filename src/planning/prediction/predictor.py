from typing import List

import numpy as np

from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.state.state import DynamicObject, EgoState, State
from mapping.src.model.map_api import MapAPI


class Predictor:
    """
    Base class for predictors of dynamic objects
    """

    @staticmethod
    def predict_object_trajectory(dynamic_object: DynamicObject, predicted_timestamps: np.ndarray, map_api: MapAPI,
                                  nav_plan:NavigationPlanMsg) -> np.ndarray:
        """
        Method to compute future locations, yaw, and velocities for dynamic objects. Returns the np.array used by the
         trajectory planner.
        :param dynamic_object: in map coordinates
        :param predicted_timestamps: np array of timestamps to predict_object_trajectory for. In ascending order.
        :param map_api: used in order to get the predicted trajectory and center lanes in map  coordinates
        :param nav_plan: predicted navigation plan of the object
        :return: predicted object's locations in global map coordinates np.array([x, y, theta, vel])
        """
        pass


    @staticmethod
    def predict_object_state(dynamic_object: DynamicObject, predicted_timestamps: np.ndarray, map_api: MapAPI,
                                  nav_plan:NavigationPlanMsg) -> List[DynamicObject]:
        """
        Wrapper method that uses the predict_object_trajectory, and creates the dynamic object list.
        :param dynamic_object: in map coordinates
        :param predicted_timestamps: np array of timestamps to predict_object_trajectory for. In ascending order.
        :param map_api: used in order to get the predicted trajectory and center lanes in map  coordinates
        :param nav_plan: predicted navigation plan of the object
        :return: List of dynamic objects whose pos/yaw/vel values are predicted using predict_object_trajectory.
        """
        pass


    @staticmethod
    def predict_ego_state(ego_state: EgoState, predicted_timestamps: np.ndarray, map_api: MapAPI,
                                  nav_plan:NavigationPlanMsg) -> List[EgoState]:
        pass


    @staticmethod
    def predict_state(state: State, predicted_timestamps: np.ndarray, map_api: MapAPI,
                          nav_plan: NavigationPlanMsg) -> List[State]:
        """
         Wrapper method that uses the predict_ego_state and predict_object_state, and creates a list containing the
         complete predicted states. TODO - consider adding reference route so that this method will be able to project
         the current state to the reference route, for example to a different lane.
        :param state: State object
        :param predicted_timestamps: np array of timestamps to predict_object_trajectory for. In ascending order.
        :param map_api: used in order to get the predicted trajectory and center lanes in map  coordinates
        :param nav_plan: predicted navigation plan of the object
        :return: a list of predicted states.
        """
        pass