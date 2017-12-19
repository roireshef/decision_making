from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import DynamicObject, ObjectSize, EgoState, State, OccupancyState
from decision_making.src.state.state_module import StateModule
import numpy as np

from decision_making.test.constants import MAP_SERVICE_ABSOLUTE_PATH
from mapping.src.model.localization import RoadLocalization
from mapping.test.model.testable_map_fixtures import map_api_mock
from rte.python.logger.AV_logger import AV_Logger

from unittest.mock import patch


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_predictObjectTrajectories_precisePrediction():
    logger = AV_Logger.get_logger("test_predictObjectTrajectories_precisePrediction")
    predictor = RoadFollowingPredictor(logger)
    size = ObjectSize(1, 1, 1)
    global_pos = np.array([500.0, 0.0, 0.0])
    dyn_obj = DynamicObject(obj_id=1, timestamp=1e9, x=global_pos[0], y=global_pos[1], z=0, yaw=0, size=size, confidence=0,
                            v_x = 10, v_y = 0, acceleration_lon=0, omega_yaw=0)
    pred_timestamps = np.arange(5.0, 12.0, 0.1)
    traj = predictor.predict_object(dyn_obj, pred_timestamps)
    assert np.isclose(traj[0][0], 540.) and np.isclose(traj[0][1], 0.) and \
           np.isclose(traj[-1][0], 600.) and np.isclose(traj[-1][1], 9.)

    dyn_obj.v_x = 0
    traj = predictor.predict_object(dyn_obj, pred_timestamps)
    assert np.isclose(traj[0][0], 500.) and np.isclose(traj[0][1], 0.) and np.isclose(traj[-1][0], 500.)


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_predictObjectOnRoad_precisePrediction():
    logger = AV_Logger.get_logger("test_predictObjectOnRoad_precisePrediction")
    predictor = RoadFollowingPredictor(logger)
    size = ObjectSize(1, 1, 1)
    global_pos = np.array([500.0, 0.0, 0.0])
    road_localization = RoadLocalization(road_id=1, lane_num=0,
                                         intra_road_lat=global_pos[1], intra_lane_lat=global_pos[1],
                                         road_lon=global_pos[0], intra_road_yaw=0)
    velocity = 10
    timestamp_in_sec = 1
    pred_timestamps = np.arange(5.0, 12.0, 0.1)
    pred_road_localizations = predictor.predict_object_on_road(road_localization, timestamp_in_sec, velocity,
                                                               pred_timestamps)
    assert np.isclose(pred_road_localizations[0].road_lon, 540.) and \
           np.isclose(pred_road_localizations[0].intra_road_lat, global_pos[1]) and \
           np.isclose(pred_road_localizations[-1].road_lon, 609.)


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_predictState_precisePrediction():
    logger = AV_Logger.get_logger("test_predictState_precisePrediction")
    predictor = RoadFollowingPredictor(logger)
    size = ObjectSize(1, 1, 1)
    dyn_global_pos = np.array([500.0, 0.0, 0.0])
    dyn_obj = DynamicObject(obj_id=1, timestamp=1e9, x=dyn_global_pos[0], y=dyn_global_pos[1], z=0, yaw=0, size=size,
                            confidence=0, v_x = 10, v_y = 0,acceleration_lon=0, omega_yaw=0)

    ego_global_pos = np.array([450.0, 0.0, 0.0])
    ego = EgoState(obj_id=0, timestamp=2e9, x=ego_global_pos[0], y=ego_global_pos[1], z=0, yaw=0, size=size,
                   confidence=0, v_x = 20, v_y = 0, acceleration_lon=0, omega_yaw=0, steering_angle=0)

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
