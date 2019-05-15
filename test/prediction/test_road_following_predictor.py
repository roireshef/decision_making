from decision_making.src.scene.scene_static_model import SceneStaticModel
from typing import List
from unittest.mock import patch
import numpy as np

from decision_making.src.planning.trajectory.samplable_trajectory import SamplableTrajectory
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.state.state import DynamicObject, State, EgoState
from decision_making.test.prediction.conftest import DYNAMIC_OBJECT_ID, CARTESIAN_CREATION
from decision_making.test.prediction.utils import Utils


def test_PredictObjects_StraightRoad_AccuratePrediction(road_following_predictor: EgoAwarePredictor,
                                                        init_state: State, prediction_timestamps: np.ndarray,
                                                        predicted_dyn_object_states_road_yaw: List[DynamicObject],
                                                        ego_samplable_trajectory: SamplableTrajectory,
                                                        scene_static_testable):

    SceneStaticModel.get_instance().set_scene_static(scene_static_testable)

    predicted_objects = road_following_predictor.predict_objects(state=init_state, object_ids=[DYNAMIC_OBJECT_ID],
                                                                 prediction_timestamps=prediction_timestamps,
                                                                 action_trajectory=ego_samplable_trajectory)

    actual_num_predictions = len(predicted_objects[DYNAMIC_OBJECT_ID])
    expected_num_predictions = len(prediction_timestamps)

    assert actual_num_predictions == expected_num_predictions

    # Verify predictions are made for the same timestamps and same order
    timestamp_ind = 0
    for actual_predicted_object in predicted_objects[DYNAMIC_OBJECT_ID]:
        assert np.isclose(actual_predicted_object.timestamp_in_sec, prediction_timestamps[timestamp_ind])
        Utils.assert_dyn_objects_numerical_fields_are_equal(actual_predicted_object,
                                                            predicted_dyn_object_states_road_yaw[timestamp_ind])
        timestamp_ind += 1


def test_PredictState_StraightRoad_AccuratePrediction(road_following_predictor: EgoAwarePredictor, init_state: State, prediction_timestamps: np.ndarray,
                                                      predicted_dyn_object_states_road_yaw: List[DynamicObject],
                                                      predicted_static_ego_states: List[EgoState],
                                                      ego_samplable_trajectory: SamplableTrajectory,
                                                      scene_static_testable):

    SceneStaticModel.get_instance().set_scene_static(scene_static_testable)

    predicted_states = road_following_predictor.predict_state(state=init_state,
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
        Utils.assert_dyn_objects_numerical_fields_are_equal(actual_predicted_object,
                                                        predicted_dyn_object_states_road_yaw[timestamp_ind])
        actual_predicted_ego = predicted_state.ego_state
        assert np.isclose(actual_predicted_ego.timestamp_in_sec, prediction_timestamps[timestamp_ind])
        Utils.assert_dyn_objects_numerical_fields_are_equal(actual_predicted_ego,
                                                        predicted_static_ego_states[timestamp_ind])
        timestamp_ind += 1


def test_PredictObjects_StraightRoad_NoCartesian(road_following_predictor: EgoAwarePredictor,
                                                 init_state: State, prediction_timestamps: np.ndarray,
                                                 predicted_dyn_object_states_road_yaw: List[DynamicObject],
                                                 ego_samplable_trajectory: SamplableTrajectory,
                                                 scene_static_testable):

    SceneStaticModel.get_instance().set_scene_static(scene_static_testable)

    with patch(CARTESIAN_CREATION) as cartesian_creation_mock:
        _ = road_following_predictor.predict_objects(state=init_state, object_ids=[DYNAMIC_OBJECT_ID],
                                                     prediction_timestamps=prediction_timestamps,
                                                     action_trajectory=ego_samplable_trajectory)

        cartesian_creation_mock.assert_not_called()
