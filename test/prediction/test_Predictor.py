import numpy as np
from typing import Type

from decision_making.src.prediction.predictor import Predictor

from decision_making.src.prediction.columns import PREDICT_X, PREDICT_Y, PREDICT_YAW, PREDICT_VEL
from decision_making.src.state.state import DynamicObject, EgoState, State
from decision_making.test.planning.custom_fixtures import state
from mapping.test.model.testable_map_fixtures import testable_map_api
from rte.python.logger.AV_logger import AV_Logger


class TestPredictorMock(Predictor):
    def predict_object(self, dynamic_object: Type[DynamicObject],
                       prediction_timestamps: np.ndarray) -> np.ndarray:
        traj = np.array([[0.0, 0.0, np.pi / 4, x] for x in range(len(prediction_timestamps))])
        traj[:, PREDICT_X] = np.cumsum(traj[:, PREDICT_VEL] * np.cos(traj[:, PREDICT_YAW]))
        traj[:, PREDICT_Y] = np.cumsum(traj[:, PREDICT_VEL] * np.sin(traj[:, PREDICT_YAW]))

        return traj


def test_predictEgoState_apiTest_returnsEgoStatesList(state, testable_map_api):
    ego_state = state.ego_state
    logger = AV_Logger.get_logger("test_predictEgoState_apiTest_returnsEgoStatesList")
    predicted_timestamps = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    test_predictor_mock = TestPredictorMock(map_api=testable_map_api, logger=logger)
    predicted_states = test_predictor_mock._predict_ego_state(ego_state=ego_state,
                                                              prediction_timestamps=predicted_timestamps)

    assert np.all([isinstance(predicted_states[x], EgoState) for x in range(len(predicted_states))])
    assert len(predicted_states) == len(predicted_timestamps)


def test_predictState_apiTest_returnsStatesList(state, testable_map_api):
    predicted_timestamps = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    logger = AV_Logger.get_logger("test_predictEgoState_apiTest_returnsEgoStatesList")
    test_predictor_mock = TestPredictorMock(map_api=testable_map_api, logger=logger)
    predicted_states = test_predictor_mock.predict_state(state,
                                                         prediction_timestamps=predicted_timestamps)

    assert np.all([isinstance(predicted_states[x], State) for x in range(len(predicted_states))])
    assert len(predicted_states) == len(predicted_timestamps)
    for predicted_state in predicted_states:
        assert np.all([isinstance(predicted_state.dynamic_objects[x], DynamicObject) for x in
                       range(len(predicted_state.dynamic_objects))])
