import copy
from typing import List
from unittest.mock import patch

import numpy as np
import yaml

from decision_making.src.prediction.action_unaware_prediction.action_unaware_predictor import ActionUnawarePredictor
from decision_making.src.state.state import DynamicObject, State
from decision_making.test.constants import MAP_SERVICE_ABSOLUTE_PATH
from decision_making.test.prediction.fixtures import DYNAMIC_OBJECT_ID
from mapping.test.model.testable_map_fixtures import map_api_mock
from decision_making.test.prediction.fixtures import road_action_unaware_predictor, init_state, prediction_timestamps, \
    predicted_dyn_object_states_road_yaw


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_AlignObjects_CurvedRoad_AccuratePrediction(road_action_unaware_predictor: ActionUnawarePredictor,
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
        assert np.isclose(actual_predicted_object.timestamp, prediction_timestamps[timestamp_ind])
        timestamp_ind += 1

        assert yaml.dump(actual_predicted_object) == yaml.dump(predicted_dyn_object_states_road_yaw[timestamp_ind])
