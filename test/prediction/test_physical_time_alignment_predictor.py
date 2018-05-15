from typing import List
from unittest.mock import patch

import numpy as np
from networkx.algorithms.centrality import current_flow_betweenness

from decision_making.src.prediction.time_alignment_prediction.time_alignment_predictor import TimeAlignmentPredictor
from decision_making.src.state.state import DynamicObject, EgoState, State
from decision_making.test.constants import MAP_SERVICE_ABSOLUTE_PATH
from decision_making.test.prediction.utils import Utils
from mapping.test.model.testable_map_fixtures import map_api_mock

from decision_making.test.prediction.conftest import physical_time_alignment_predictor, init_state, \
    prediction_timestamps, \
    predicted_dyn_object_states_road_yaw, DYNAMIC_OBJECT_ID


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_AlignObjects_ExternalTimestamp_AccuratePrediction(physical_time_alignment_predictor: TimeAlignmentPredictor,
                                                           init_state: State, prediction_timestamps: np.ndarray,
                                                           predicted_dyn_object_states_constant_yaw: List[DynamicObject],
                                                           predicted_static_ego_states: List[EgoState]):
    predicted_state = physical_time_alignment_predictor.align_objects_to_most_recent_timestamp(state=init_state,
                                                                                               current_timestamp=
                                                                                               prediction_timestamps[0])

    # Verify predictions are made for the same timestamps and same order
    actual_predicted_object = predicted_state.get_object_from_state(predicted_state, DYNAMIC_OBJECT_ID)
    assert np.isclose(actual_predicted_object.timestamp_in_sec, prediction_timestamps[0])
    Utils.assert_objects_numerical_fields_are_equal(actual_predicted_object,
                                                    predicted_dyn_object_states_constant_yaw[0])
    actual_predicted_ego = predicted_state.ego_state
    assert np.isclose(actual_predicted_ego.timestamp_in_sec, prediction_timestamps[0])
    Utils.assert_objects_numerical_fields_are_equal(actual_predicted_ego,
                                                    predicted_static_ego_states[0])

@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_AlignObjects_ExternalTimestamp_ConstantYawAccuratePrediction(physical_time_alignment_predictor: TimeAlignmentPredictor,
                                                           init_state: State, prediction_timestamps: np.ndarray,
                                                           predicted_dyn_object_states_constant_yaw: List[DynamicObject],
                                                           predicted_static_ego_states: List[EgoState]):
    predicted_state = physical_time_alignment_predictor.align_objects_to_most_recent_timestamp(state=init_state,
                                                                                               current_timestamp=
                                                                                               prediction_timestamps[1])

    # Verify predictions are made for the same timestamps and same order
    actual_predicted_object = predicted_state.get_object_from_state(predicted_state, DYNAMIC_OBJECT_ID)
    assert np.isclose(actual_predicted_object.timestamp_in_sec, prediction_timestamps[1])
    Utils.assert_objects_numerical_fields_are_equal(actual_predicted_object,
                                                    predicted_dyn_object_states_constant_yaw[1])
    actual_predicted_ego = predicted_state.ego_state
    assert np.isclose(actual_predicted_ego.timestamp_in_sec, prediction_timestamps[1])
    Utils.assert_objects_numerical_fields_are_equal(actual_predicted_ego,
                                                    predicted_static_ego_states[1])


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_AlignObjects_NoTimestamp_AccuratePrediction(physical_time_alignment_predictor: TimeAlignmentPredictor,
                                                     unaligned_state: State,
                                                     aligned_ego_state: EgoState):
    predicted_state = physical_time_alignment_predictor.align_objects_to_most_recent_timestamp(state=unaligned_state)

    # Verify predictions are made for the same timestamps and same order
    actual_predicted_object = predicted_state.get_object_from_state(predicted_state, DYNAMIC_OBJECT_ID)
    expected_predicted_object = unaligned_state.get_object_from_state(unaligned_state,DYNAMIC_OBJECT_ID)
    assert np.isclose(actual_predicted_object.timestamp_in_sec,
                      expected_predicted_object.timestamp_in_sec)
    Utils.assert_objects_numerical_fields_are_equal(actual_predicted_object,
                                                    expected_predicted_object)
    actual_predicted_ego = predicted_state.ego_state
    assert np.isclose(actual_predicted_ego.timestamp_in_sec, aligned_ego_state.timestamp_in_sec)
    Utils.assert_objects_numerical_fields_are_equal(actual_predicted_ego,
                                                    aligned_ego_state)
