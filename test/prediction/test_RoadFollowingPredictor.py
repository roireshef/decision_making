import copy
from unittest.mock import patch

import numpy as np

from decision_making.src.global_constants import DEFAULT_OBJECT_Z_VALUE
from decision_making.src.planning.types import C_Y, C_X, C_V, C_YAW
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import DynamicObject, EgoState, State, OccupancyState
from decision_making.test.constants import MAP_SERVICE_ABSOLUTE_PATH
from mapping.test.model.testable_map_fixtures import map_api_mock
from rte.python.logger.AV_logger import AV_Logger


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_predictObjectTrajectories_precisePredictionDynamicAndStaticObjectMultipleSingleTimestamp(car_size):
    logger = AV_Logger.get_logger("test_predictObjectTrajectories_precisePrediction")
    predictor = RoadFollowingPredictor(logger)
    global_state = np.array([500.0, 0.0, 0.0, 10.0])
    dyn_obj = DynamicObject(obj_id=1, timestamp=1e9, x=global_state[C_X], y=global_state[C_Y], z=DEFAULT_OBJECT_Z_VALUE,
                            yaw=global_state[C_YAW], size=car_size, confidence=0, v_x=global_state[C_V], v_y=0,
                            acceleration_lon=0, omega_yaw=0)
    # test for dynamic object with multiple timestamps
    pred_timestamps = np.arange(5.0, 12.0, 0.1)
    traj = predictor.predict_object(dyn_obj, pred_timestamps)
    assert np.isclose(traj[0][C_X], 540.) and np.isclose(traj[0][C_Y], 0.) and \
           np.isclose(traj[-1][C_X], 600.) and np.isclose(traj[-1][C_Y], 9.)

    # test for static object
    stat_obj = copy.deepcopy(dyn_obj)
    stat_obj.v_x = 0
    traj = predictor.predict_object(stat_obj, pred_timestamps)
    assert np.isclose(traj[0][C_X], 500.) and np.isclose(traj[0][C_Y], 0.) and np.isclose(traj[-1][C_X], 500.)

    # test for single prediction timestamp
    traj = predictor.predict_object(dyn_obj, np.array([pred_timestamps[0]]))
    assert np.isclose(traj[0][C_X], 540.) and np.isclose(traj[0][C_Y], 0.)


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_predictObject_zeroSpeedZeroLookahead_samePoint(car_size):
    logger = AV_Logger.get_logger("test_predictObjectTrajectories_precisePrediction")
    predictor = RoadFollowingPredictor(logger)
    global_state = np.array([500.0, 0.0, 0.0, 0.0])
    dyn_obj = DynamicObject(obj_id=1, timestamp=0, x=global_state[C_X], y=global_state[C_Y], z=DEFAULT_OBJECT_Z_VALUE,
                            yaw=global_state[C_YAW], size=car_size, confidence=0, v_x=global_state[C_V], v_y=0.0,
                            acceleration_lon=0, omega_yaw=0)
    # Test if zero lookahead at zero speed works without raising exception
    pred_timestamps = np.array([0.0])
    predicted_traj = predictor.predict_object(dyn_obj, pred_timestamps)

    assert np.any(np.isclose(global_state, predicted_traj))


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_predictState_precisePrediction(car_size):
    logger = AV_Logger.get_logger("test_predictState_precisePrediction")
    predictor = RoadFollowingPredictor(logger)
    dyn_global_state = np.array([500.0, 0.0, 0.0, 10.0])
    dyn_obj = DynamicObject(obj_id=1, timestamp=1e9, x=dyn_global_state[C_X], y=dyn_global_state[C_Y],
                            z=DEFAULT_OBJECT_Z_VALUE, yaw=dyn_global_state[C_YAW], size=car_size,
                            confidence=0, v_x=dyn_global_state[C_V], v_y=0, acceleration_lon=0, omega_yaw=0)

    ego_global_pos = np.array([450.0, 0.0, 0.0, 20.0])
    ego = EgoState(obj_id=0, timestamp=2e9, x=ego_global_pos[C_X], y=ego_global_pos[C_Y], z=DEFAULT_OBJECT_Z_VALUE,
                   yaw=ego_global_pos[C_YAW], size=car_size, confidence=0, v_x=ego_global_pos[C_V], v_y=0,
                   acceleration_lon=0, omega_yaw=0, steering_angle=0)

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
