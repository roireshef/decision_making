from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.test.messages.scene_static_fixture import scene_static_testable
from typing import List

import numpy as np

from decision_making.src.prediction.action_unaware_prediction.ego_unaware_predictor import EgoUnawarePredictor
from decision_making.src.state.state import DynamicObject, EgoState, State
from decision_making.test.prediction.utils import Utils

from decision_making.test.prediction.conftest import physical_time_alignment_predictor, dynamic_init_state, \
    prediction_timestamps, static_cartesian_state, \
    predicted_dyn_object_states_road_yaw, predicted_dynamic_ego_states, DYNAMIC_OBJECT_ID


def test_AlignObjects_ExternalTimestamp_AccuratePrediction(physical_time_alignment_predictor: EgoUnawarePredictor,
                                                           dynamic_init_state: State, prediction_timestamps: np.ndarray,
                                                           predicted_dyn_object_states_constant_yaw: List[
                                                               DynamicObject],
                                                           predicted_dynamic_ego_states: List[EgoState],
                                                           scene_static_testable):

    SceneStaticModel.get_instance().set_scene_static(scene_static_testable)

    predicted_state = physical_time_alignment_predictor.predict_state(state=dynamic_init_state, prediction_timestamps=np.array([
        prediction_timestamps[0]]))[0]

    # Verify predictions are made for the same timestamps and same order
    actual_predicted_object = predicted_state.get_object_from_state(predicted_state, DYNAMIC_OBJECT_ID)
    assert np.isclose(actual_predicted_object.timestamp_in_sec, prediction_timestamps[0])
    Utils.assert_dyn_objects_numerical_fields_are_equal(actual_predicted_object,
                                                        predicted_dyn_object_states_constant_yaw[0])
    actual_predicted_ego = predicted_state.ego_state
    assert np.isclose(actual_predicted_ego.timestamp_in_sec, prediction_timestamps[0])
    Utils.assert_dyn_objects_numerical_fields_are_equal(actual_predicted_ego,
                                                        predicted_dynamic_ego_states[0])


def test_AlignObjects_ExternalTimestamp_ConstantYawAccuratePrediction(
        physical_time_alignment_predictor: EgoUnawarePredictor,
        dynamic_init_state: State, prediction_timestamps: np.ndarray,
        predicted_dyn_object_states_constant_yaw: List[DynamicObject],
        predicted_dynamic_ego_states: List[EgoState],
        scene_static_testable):

    SceneStaticModel.get_instance().set_scene_static(scene_static_testable)

    predicted_state = physical_time_alignment_predictor.predict_state(state=dynamic_init_state, prediction_timestamps=np.array([
        prediction_timestamps[1]]))[0]

    # Verify predictions are made for the same timestamps and same order
    actual_predicted_object = predicted_state.get_object_from_state(predicted_state, DYNAMIC_OBJECT_ID)
    assert np.isclose(actual_predicted_object.timestamp_in_sec, prediction_timestamps[1])
    Utils.assert_dyn_objects_numerical_fields_are_equal(actual_predicted_object,
                                                        predicted_dyn_object_states_constant_yaw[1])
    actual_predicted_ego = predicted_state.ego_state
    assert np.isclose(actual_predicted_ego.timestamp_in_sec, prediction_timestamps[1])
    Utils.assert_dyn_objects_numerical_fields_are_equal(actual_predicted_ego,
                                                        predicted_dynamic_ego_states[1])

