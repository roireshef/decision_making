import numpy as np
from typing import Type

from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.prediction.predictor import Predictor

from decision_making.src.prediction.columns import PREDICT_X, PREDICT_Y, PREDICT_YAW, PREDICT_VEL
from decision_making.src.state.state import DynamicObject, EgoState, State
from decision_making.test.planning.custom_fixtures import state_fix, navigation_plan


class TestPredictorMock(Predictor):

    def predict_object_trajectories(self, dynamic_object: Type[DynamicObject], prediction_timestamps: np.ndarray,
                                    nav_plan: NavigationPlanMsg) -> np.ndarray:
        traj = np.array([[0.0, 0.0, np.pi / 4, x] for x in range(len(prediction_timestamps))])
        traj[:, PREDICT_X] = np.cumsum(traj[:, PREDICT_VEL] * np.cos(traj[:, PREDICT_YAW]))
        traj[:, PREDICT_Y] = np.cumsum(traj[:, PREDICT_VEL] * np.sin(traj[:, PREDICT_YAW]))

        return traj


def test_predictEgoState_apiTest_returnsEgoStatesList(state_fix, navigation_plan):
    ego_state = state_fix.ego_state

    predicted_timestamps = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    test_predictor_mock = TestPredictorMock(map_api=None)
    predicted_states = test_predictor_mock._predict_ego_state(ego_state=ego_state,
                                                            prediction_timestamps=predicted_timestamps,
                                                            nav_plan=navigation_plan)

    assert np.all([isinstance(predicted_states[x], EgoState) for x in range(len(predicted_states))])
    assert len(predicted_states) == len(predicted_timestamps)

def test_predictState_apiTest_returnsStatesList(state_fix, navigation_plan):
    predicted_timestamps = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    test_predictor_mock = TestPredictorMock(map_api=None)
    predicted_states = test_predictor_mock.predict_state(state_fix,
                                          prediction_timestamps=predicted_timestamps,
                                          nav_plan=navigation_plan)

    assert np.all([isinstance(predicted_states[x], State) for x in range(len(predicted_states))])
    assert len(predicted_states) == len(predicted_timestamps)
    for predicted_state in predicted_states:
        assert np.all([isinstance(predicted_state.dynamic_objects[x], DynamicObject) for x in range(len(predicted_state.dynamic_objects))])

