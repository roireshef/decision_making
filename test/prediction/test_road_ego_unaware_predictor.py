import copy
from typing import List
from unittest.mock import patch

import numpy as np
import yaml

from decision_making.src.prediction.action_unaware_prediction.ego_unaware_predictor import EgoUnawarePredictor
from decision_making.src.state.state import DynamicObject, State
from decision_making.test.constants import MAP_SERVICE_ABSOLUTE_PATH
from decision_making.test.prediction.conftest import DYNAMIC_OBJECT_ID
from decision_making.test.prediction.utils import Utils
from mapping.test.model.testable_map_fixtures import map_api_mock
from decision_making.test.prediction.conftest import road_action_aware_predictor, init_state, prediction_timestamps, \
    predicted_dyn_object_states_road_yaw


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_PredictObjects_CurvedRoad_AccuratePrediction(road_action_unaware_predictor: EgoUnawarePredictor,
                                                      init_state: State, prediction_timestamps: np.ndarray,
                                                      predicted_dyn_object_states_road_yaw: List[DynamicObject]):
    predicted_objects = road_action_unaware_predictor.predict_objects(state=init_state, object_ids=[DYNAMIC_OBJECT_ID],
                                                                      prediction_timestamps=prediction_timestamps)

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
