import numpy as np
from typing import Type

from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.prediction.predictor import Predictor

from decision_making.src.prediction.columns import PREDICT_X, PREDICT_Y, PREDICT_YAW, PREDICT_VEL
from decision_making.src.state.state import DynamicObject, EgoState
from decision_making.test.planning.custom_fixtures import state, navigation_plan
from mapping.src.model.map_api import MapAPI


class TestPredictorMock(Predictor):
    @classmethod
    def predict_object_trajectories(cls, dynamic_object: Type[DynamicObject], prediction_timestamps: np.ndarray,
                                    map_api: MapAPI,
                                    nav_plan: NavigationPlanMsg) -> np.ndarray:
        traj = np.array([[0.0, 0.0, np.pi / 4, x] for x in range(len(prediction_timestamps))])
        traj[:, PREDICT_X] = np.cumsum(traj[:, PREDICT_VEL] * np.cos(traj[:, PREDICT_YAW]))
        traj[:, PREDICT_Y] = np.cumsum(traj[:, PREDICT_VEL] * np.sin(traj[:, PREDICT_YAW]))

        return traj


def test_predictEgoState_apiTest_returnsEgoStatesList(state, navigation_plan):
    ego_state = state.ego_state

    predicted_timestamps = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    predicted_states = TestPredictorMock._predict_ego_state(ego_state=ego_state,
                                                            prediction_timestamps=predicted_timestamps, map_api=None,
                                                            nav_plan=navigation_plan)

    assert np.all([isinstance(predicted_states[x], EgoState) for x in range(len(predicted_states))])
    assert len(predicted_states) == len(predicted_timestamps)
