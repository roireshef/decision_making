from typing import List
from unittest.mock import patch
import numpy as np

from decision_making.src.planning.trajectory.trajectory_planner import SamplableTrajectory
from decision_making.src.state.state import NewDynamicObject, State, NewEgoState
from decision_making.test.constants import MAP_SERVICE_ABSOLUTE_PATH
from decision_making.test.prediction.conftest import constant_velocity_predictor, init_state, prediction_timestamps, \
    predicted_dyn_object_states_road_yaw, ego_samplable_trajectory, static_cartesian_state, \
    predicted_static_ego_states, static_cartesian_state, DYNAMIC_OBJECT_ID
from decision_making.test.prediction.utils import Utils
from mapping.test.model.testable_map_fixtures import map_api_mock
from decision_making.src.prediction.ego_aware_prediction.maneuver_based_predictor import ManeuverBasedPredictor

@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_PredictObjects_StraightRoad_AccuratePrediction(constant_velocity_predictor: ManeuverBasedPredictor,
                                                      init_state: State, prediction_timestamps: np.ndarray,
                                                      predicted_dyn_object_states_road_yaw: List[NewDynamicObject],
                                                      ego_samplable_trajectory: SamplableTrajectory):

    predicted_objects = constant_velocity_predictor.predict_objects(state=init_state, object_ids=[DYNAMIC_OBJECT_ID],
                                                  prediction_timestamps=prediction_timestamps,
                                                  action_trajectory=ego_samplable_trajectory)

    actual_num_predictions = len(predicted_objects[DYNAMIC_OBJECT_ID])
    expected_num_predictions = len(prediction_timestamps)

    assert actual_num_predictions == expected_num_predictions

    # Verify predictions are made for the same timestamps and same order
    timestamp_ind = 0
    for actual_predicted_object in predicted_objects[DYNAMIC_OBJECT_ID]:
        assert np.isclose(actual_predicted_object.timestamp_in_sec, prediction_timestamps[timestamp_ind])
        Utils.assert_objects_numerical_fields_are_equal(actual_predicted_object,
                                                        predicted_dyn_object_states_road_yaw[timestamp_ind])
        timestamp_ind += 1


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_PredictState_StraightRoad_AccuratePrediction(constant_velocity_predictor: ManeuverBasedPredictor, init_state: State, prediction_timestamps: np.ndarray,
                                                    predicted_dyn_object_states_road_yaw: List[NewDynamicObject],
                                                    predicted_static_ego_states: List[NewEgoState],
                                                    ego_samplable_trajectory: SamplableTrajectory):

    predicted_states = constant_velocity_predictor.predict_state(state=init_state,
                                               prediction_timestamps=prediction_timestamps,
                                               action_trajectory=ego_samplable_trajectory)

    actual_num_predictions = len(predicted_states)
    expected_num_predictions = len(prediction_timestamps)

    assert actual_num_predictions == expected_num_predictions

    # Verify predictions are made for the same timestamps and same order
    timestamp_ind = 0
    for predicted_state in predicted_states:
        actual_predicted_object = predicted_state.get_object_from_state(predicted_state, DYNAMIC_OBJECT_ID)
        assert np.isclose(actual_predicted_object.timestamp_in_sec, prediction_timestamps[timestamp_ind])
        Utils.assert_objects_numerical_fields_are_equal(actual_predicted_object,
                                                        predicted_dyn_object_states_road_yaw[timestamp_ind])
        actual_predicted_ego = predicted_state.ego_state
        assert np.isclose(actual_predicted_ego.timestamp_in_sec, prediction_timestamps[timestamp_ind])
        Utils.assert_objects_numerical_fields_are_equal(actual_predicted_ego,
                                                        predicted_static_ego_states[timestamp_ind])
        timestamp_ind += 1
