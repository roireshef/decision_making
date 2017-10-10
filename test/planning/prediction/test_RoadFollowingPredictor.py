import pytest
import numpy as np

from decision_making.src.state.state import DynamicObject, ObjectSize
from decision_making.src.planning.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state_module import StateModule
from mapping.test.model.testable_map_fixtures import *


def test_predict_movingCarAlongRoad_precisePrediction(testable_map_api, navigation_fixture):
    map_api = testable_map_api
    nav_plan = navigation_fixture

    predictor = RoadFollowingPredictor()
    size = ObjectSize(1, 1, 1)
    global_pos = np.array([500.0, 0.0, 0.0])
    road_localization = StateModule._compute_road_localization(global_pos=global_pos, global_yaw=0, map_api=map_api)
    dyn_obj = DynamicObject(obj_id=1, timestamp=0, x=global_pos[0], y=global_pos[1], z=0, yaw=0, size=size, confidence=0,
                            v_x = 10, v_y = 0,
                            acceleration_lon=0, omega_yaw=0, road_localization=road_localization)
    pred_timestamps = np.arange(5.0, 12.0, 0.1)
    traj = predictor.predict_object_trajectory(dyn_obj, pred_timestamps, map_api, nav_plan)
    assert np.isclose(traj[0][0],550.) and np.isclose(traj[0][1], 0.) and \
           np.isclose(traj[-1][0], 600.) and np.isclose(traj[-1][1], 19.)
