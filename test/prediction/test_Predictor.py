from typing import Type, List
from unittest.mock import patch

import numpy as np

from decision_making.src.planning.types import C_X, C_Y, C_THETA, C_V
from decision_making.src.planning.types import CartesianTrajectory
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import DynamicObject, EgoState, State
from decision_making.test.constants import MAP_SERVICE_ABSOLUTE_PATH
from mapping.test.model.testable_map_fixtures import testable_map_api
from decision_making.test.planning.custom_fixtures import state
from rte.python.logger.AV_logger import AV_Logger

from mapping.test.model.testable_map_fixtures import testable_map_api
from mapping.src.model.localization import RoadLocalization


# TODO: Move to fixtures, so that other tests will be able to use it
class TestPredictorMock(Predictor):
    def predict_object(self, dynamic_object: Type[DynamicObject],
                       prediction_timestamps: np.ndarray) -> np.ndarray:
        traj: CartesianTrajectory = np.array([[0.0, 0.0, np.pi / 4, x] for x in range(len(prediction_timestamps))])
        traj[:, C_X] = np.cumsum(traj[:, C_V] * np.cos(traj[:, C_THETA]))
        traj[:, C_Y] = np.cumsum(traj[:, C_V] * np.sin(traj[:, C_THETA]))

        return traj



@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=testable_map_api)
def test_predictEgoState_apiTest_returnsEgoStatesList(state):
    ego_state = state.ego_state
    logger = AV_Logger.get_logger("test_predictEgoState_apiTest_returnsEgoStatesList")
    predicted_timestamps = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    test_predictor_mock = TestPredictorMock(logger=logger)
    predicted_states = test_predictor_mock._predict_object_state(dynamic_object=ego_state,
                                                              prediction_timestamps=predicted_timestamps)

    assert np.all([isinstance(predicted_states[x], EgoState) for x in range(len(predicted_states))])
    assert len(predicted_states) == len(predicted_timestamps)


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=testable_map_api)
def test_predictState_apiTest_returnsStatesList(state):
    predicted_timestamps = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    logger = AV_Logger.get_logger("test_predictEgoState_apiTest_returnsEgoStatesList")
    test_predictor_mock = TestPredictorMock(logger=logger)
    predicted_states = test_predictor_mock.predict_state(state,
                                                         prediction_timestamps=predicted_timestamps)

    assert np.all([isinstance(predicted_states[x], State) for x in range(len(predicted_states))])
    assert len(predicted_states) == len(predicted_timestamps)
    for predicted_state in predicted_states:
        assert np.all([isinstance(predicted_state.dynamic_objects[x], DynamicObject) for x in
                       range(len(predicted_state.dynamic_objects))])

