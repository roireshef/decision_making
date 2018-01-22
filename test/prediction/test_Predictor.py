from typing import Type, List
from unittest.mock import patch

import numpy as np

from decision_making.src.state.state import DynamicObject, EgoState, State
from decision_making.test.constants import MAP_SERVICE_ABSOLUTE_PATH
from decision_making.test.prediction.mock_predictor import TestPredictorMock
from mapping.test.model.testable_map_fixtures import testable_map_api
from decision_making.test.planning.custom_fixtures import state
from rte.python.logger.AV_logger import AV_Logger

from mapping.test.model.testable_map_fixtures import testable_map_api


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


def test_alignObjectsToMostRecentTimestamp_futureTimestamp_allObjectsAligned(state):
    logger = AV_Logger.get_logger("test_alignObjectsToMostRecentTimestamp_futureTimestamp_allObjectsAligned")
    objects_timestamps = np.array([x.timestamp for x in state.dynamic_objects])
    objects_timestamps = np.r_[objects_timestamps, state.ego_state.timestamp]
    max_timestamp = int(np.max(objects_timestamps, state.ego_state.timestamp))
    future_timestamp = max_timestamp + 1000

    test_predictor_mock = TestPredictorMock(logger=logger)
    aligned_state = test_predictor_mock.align_objects_to_most_recent_timestamp(state,
                                                                               current_timestamp=future_timestamp*1E-9)

    aligned_timestamps = np.array([x.timestamp for x in aligned_state.dynamic_objects])
    aligned_timestamps = np.r_[aligned_timestamps, aligned_state.ego_state.timestamp]

    assert np.all(np.isclose(aligned_timestamps, future_timestamp))


def test_alignObjectsToMostRecentTimestamp_noFutureTimestamp_allObjectsAligned(state):
    logger = AV_Logger.get_logger("test_alignObjectsToMostRecentTimestamp_futureTimestamp_allObjectsAligned")
    objects_timestamps = np.array([x.timestamp for x in state.dynamic_objects])
    objects_timestamps = np.r_[objects_timestamps, state.ego_state.timestamp]
    max_timestamp = int(np.max(objects_timestamps, state.ego_state.timestamp))

    test_predictor_mock = TestPredictorMock(logger=logger)
    aligned_state = test_predictor_mock.align_objects_to_most_recent_timestamp(state)

    aligned_timestamps = np.array([x.timestamp for x in aligned_state.dynamic_objects])
    aligned_timestamps = np.r_[aligned_timestamps, aligned_state.ego_state.timestamp]

    assert np.all(np.isclose(aligned_timestamps, max_timestamp))
