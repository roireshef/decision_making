from typing import List, Type

import copy
import numpy as np

from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.prediction.columns import PREDICT_X, PREDICT_Y, PREDICT_YAW, PREDICT_VEL
from decision_making.src.state.state import DynamicObject, EgoState, State, ObjectSize, RoadLocalization
from mapping.src.model.map_api import MapAPI


class Predictor:
    """
    Base class for predictors of dynamic objects
    """

    @classmethod
    def predict_object_trajectory(cls, dynamic_object: Type[DynamicObject], predicted_timestamps: np.ndarray,
                                  map_api: MapAPI,
                                  nav_plan: NavigationPlanMsg) -> np.ndarray:
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

    @classmethod
    def predict_object_state(cls, dynamic_object: Type[DynamicObject], predicted_timestamps: np.ndarray,
                             map_api: MapAPI,
                             nav_plan: NavigationPlanMsg) -> List[Type[DynamicObject]]:
        """
        Wrapper method that uses the predict_object_trajectory, and creates the dynamic object list.
        :param dynamic_object: in map coordinates
        :param predicted_timestamps: np array of timestamps to predict_object_trajectory for. In ascending order.
        :param map_api: used in order to get the predicted trajectory and center lanes in map  coordinates
        :param nav_plan: predicted navigation plan of the object
        :return: List of dynamic objects whose pos/yaw/vel values are predicted using predict_object_trajectory.
        """

        predicted_object_trajectory = cls.predict_object_trajectory(dynamic_object, predicted_timestamps, map_api,
                                                                    nav_plan)

        # Initiate array of DynamicObject at predicted times
        predicted_object_states = [copy.deepcopy(dynamic_object) for x in range(len(predicted_timestamps))]

        # Fill with predicted state
        for t_ind, predicted_object_state in enumerate(predicted_object_states):
            predicted_object_state.timestamp = predicted_timestamps[t_ind]
            predicted_object_state.x = predicted_object_trajectory[t_ind, PREDICT_X]
            predicted_object_state.y = predicted_object_trajectory[t_ind, PREDICT_Y]
            predicted_object_state.yaw = predicted_object_trajectory[t_ind, PREDICT_YAW]
            # TODO: check consistency between diff of (x,y) and v_x, v_y as calculated below:
            predicted_object_state.v_x = predicted_object_trajectory[t_ind, PREDICT_VEL] * np.cos(
                predicted_object_trajectory[t_ind, PREDICT_YAW])
            predicted_object_state.v_y = predicted_object_trajectory[t_ind, PREDICT_VEL] * np.sin(
                predicted_object_trajectory[t_ind, PREDICT_YAW])

        return predicted_object_states

    @classmethod
    def predict_ego_state(cls, ego_state: EgoState, predicted_timestamps: np.ndarray, map_api: MapAPI,
                          nav_plan: NavigationPlanMsg) -> List[Type[DynamicObject]]:
        # TODO: update EgoState attributes that are copied and are not part of DynamicObject (as steering_angle)
        return cls.predict_object_state(ego_state, predicted_timestamps, map_api, nav_plan)

    @classmethod
    def predict_state(cls, state: State, predicted_timestamps: np.ndarray, map_api: MapAPI,
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

