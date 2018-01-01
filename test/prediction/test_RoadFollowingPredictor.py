import copy
from unittest.mock import patch

import numpy as np

from decision_making.test.planning.custom_fixtures import car_size
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import DynamicObject, EgoState, State, OccupancyState
from decision_making.test.constants import MAP_SERVICE_ABSOLUTE_PATH
from mapping.test.model.testable_map_fixtures import map_api_mock
from rte.python.logger.AV_logger import AV_Logger


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_predictObjectTrajectories_precisePredictionDynamicAndStaticObjectMultipleSingleTimestamp(car_size):
    logger = AV_Logger.get_logger("test_predictObjectTrajectories_precisePrediction")
    predictor = RoadFollowingPredictor(logger)
    global_pos = np.array([500.0, 0.0, 0.0])
    dyn_obj = DynamicObject(obj_id=1, timestamp=1e9, x=global_pos[0], y=global_pos[1], z=0, yaw=0, size=car_size,
                            confidence=0,
                            v_x=10, v_y=0, acceleration_lon=0, omega_yaw=0)
    # test for dynamic object with multiple timestamps
    pred_timestamps = np.arange(5.0, 12.0, 0.1)
    traj = predictor.predict_object(dyn_obj, pred_timestamps)
    assert np.isclose(traj[0][0], 540.) and np.isclose(traj[0][1], 0.) and \
           np.isclose(traj[-1][0], 600.) and np.isclose(traj[-1][1], 9.)

    # test for static object
    stat_obj = copy.deepcopy(dyn_obj)
    stat_obj.v_x = 0
    traj = predictor.predict_object(stat_obj, pred_timestamps)
    assert np.isclose(traj[0][0], 500.) and np.isclose(traj[0][1], 0.) and np.isclose(traj[-1][0], 500.)

    # test for single prediction timestamp
    traj = predictor.predict_object(dyn_obj, np.array([pred_timestamps[0]]))
    assert np.isclose(traj[0][0], 540.) and np.isclose(traj[0][1], 0.)



@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_predictObject_zeroSpeedZeroLookahead_noException(car_size):
    logger = AV_Logger.get_logger("test_predictObjectTrajectories_precisePrediction")
    predictor = RoadFollowingPredictor(logger)
    global_pos = np.array([500.0, 0.0, 0.0])
    dyn_obj = DynamicObject(obj_id=1, timestamp=0, x=global_pos[0], y=global_pos[1], z=0, yaw=0, size=car_size,
                            confidence=0,
                            v_x=0.0, v_y=0.0, acceleration_lon=0, omega_yaw=0)
    # Test if zero lookahead at zero speed works without raising exception
    pred_timestamps = np.array([0.0])
    traj = predictor.predict_object(dyn_obj, pred_timestamps)

    assert True


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_predictObjectOnRoad_precisePrediction():
    logger = AV_Logger.get_logger("test_predictObjectOnRoad_precisePrediction")
    predictor = RoadFollowingPredictor(logger)
    global_pos = np.array([500.0, 0.0, 0.0])
    velocity = 10
    dynamic_object = DynamicObject(obj_id=1, timestamp=0, x=global_pos[0], y=global_pos[1], z=0, yaw=0, size=car_size,
                                   confidence=0, v_x=velocity, v_y=0, acceleration_lon=0, omega_yaw=0)

    pred_timestamps = np.arange(4.0, 11.0, 0.1)
    pred_object_state = predictor.predict_object_on_road(dynamic_object, pred_timestamps)
    assert np.isclose(pred_object_state[0].road_localization.road_lon, 540.) and \
           np.isclose(pred_object_state[-1].road_localization.road_lon, 609.)


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_predictState_precisePrediction(car_size):
    logger = AV_Logger.get_logger("test_predictState_precisePrediction")
    predictor = RoadFollowingPredictor(logger)
    dyn_global_pos = np.array([500.0, 0.0, 0.0])
    dyn_obj = DynamicObject(obj_id=1, timestamp=1e9, x=dyn_global_pos[0], y=dyn_global_pos[1], z=0, yaw=0, size=car_size,
                            confidence=0, v_x=10, v_y=0, acceleration_lon=0, omega_yaw=0)

    ego_global_pos = np.array([450.0, 0.0, 0.0])
    ego = EgoState(obj_id=0, timestamp=2e9, x=ego_global_pos[0], y=ego_global_pos[1], z=0, yaw=0, size=car_size,
                   confidence=0, v_x=20, v_y=0, acceleration_lon=0, omega_yaw=0, steering_angle=0)

    occupancy_state = OccupancyState(0, np.array([]), np.array([]))
    state = State(occupancy_state=occupancy_state, dynamic_objects=[dyn_obj], ego_state=ego)

    pred_timestamps = np.arange(5.0, 12.0, 0.1)
    predicted_states = predictor.predict_state(state=state, prediction_timestamps=pred_timestamps)

    assert np.isclose(predicted_states[0].dynamic_objects[0].x, 540.) and \
           np.isclose(predicted_states[0].dynamic_objects[0].y, 0.) and \
           np.isclose(predicted_states[-1].dynamic_objects[0].x, 600.) and \
           np.isclose(predicted_states[-1].dynamic_objects[0].y, 9.)
    assert np.isclose(predicted_states[0].ego_state.x, 510.) and \
           np.isclose(predicted_states[0].ego_state.y, 0.) and \
           np.isclose(predicted_states[-1].ego_state.x, 600.) and \
           np.isclose(predicted_states[-1].ego_state.y, 48.)
