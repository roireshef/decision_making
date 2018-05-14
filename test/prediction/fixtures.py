from typing import List

import pytest

from decision_making.src.global_constants import DEFAULT_OBJECT_Z_VALUE
from decision_making.src.planning.types import CartesianState, C_X, C_Y, C_YAW, C_V
from decision_making.src.prediction.action_unaware_prediction.road_action_unaware_predictor import \
    RoadActionUnawarePredictor
from decision_making.src.prediction.time_alignment_prediction.time_alignment_predictor import TimeAlignmentPredictor
from decision_making.src.state.state import DynamicObject, ObjectSize, EgoState, State, OccupancyState
import numpy as np

from rte.python.logger.AV_logger import AV_Logger

DYNAMIC_OBJECT_ID = 1
EGO_OBJECT_ID = 0

@pytest.fixture(scope='function')
def physical_time_alignment_predictor() -> TimeAlignmentPredictor:
    logger = AV_Logger.get_logger("PREDICTOR_TEST_LOGGER")
    yield TimeAlignmentPredictor(logger)


@pytest.fixture(scope='function')
def road_action_unaware_predictor() -> RoadActionUnawarePredictor:
    logger = AV_Logger.get_logger("PREDICTOR_TEST_LOGGER")
    yield RoadActionUnawarePredictor(logger)


@pytest.fixture(scope='function')
def car_size() ->ObjectSize:
    yield ObjectSize(length=3.0, width=2.0, height=1.2)


@pytest.fixture(scope='function')
def init_cartesian_state() -> CartesianState:
    yield np.array([500.0, 0.0, 0.0, 10.0])


@pytest.fixture(scope='function')
def predicted_cartesian_state_0() -> CartesianState:
    yield np.array([510.0, 0.0, 0.0, 10.0])


@pytest.fixture(scope='function')
def predicted_cartesian_state_1() -> CartesianState:
    yield np.array([600.0, 0.0, 0.0, 10.0])


@pytest.fixture(scope='function')
def predicted_cartesian_state_2_constant_yaw() -> CartesianState:
    yield np.array([600.0, 100.0, 0.0, 10.0])


@pytest.fixture(scope='function')
def predicted_cartesian_state_2_road_yaw() -> CartesianState:
    yield np.array([600.0, 100.0, np.pi / 2, 10.0])


@pytest.fixture(scope='function')
def static_cartesian_state() -> CartesianState:
    yield np.array([0.0, 0.0, 0.0, 0.0])


@pytest.fixture(scope='function')
def prediction_timestamps() -> np.array:
    yield np.array([1.0, 10.0, 20.0])


@pytest.fixture(scope='function')
def init_dyn_obj(init_cartesian_state) -> DynamicObject:
    yield DynamicObject(obj_id=DYNAMIC_OBJECT_ID, timestamp=int(0e9), x=init_cartesian_state[C_X], y=init_cartesian_state[C_Y],
                        z=DEFAULT_OBJECT_Z_VALUE,
                        yaw=init_cartesian_state[C_YAW], size=car_size, confidence=0, v_x=init_cartesian_state[C_V],
                        v_y=0,
                        acceleration_lon=0, omega_yaw=0)


@pytest.fixture(scope='function')
def init_ego_state(static_cartesian_state) -> EgoState:
    yield EgoState(obj_id=EGO_OBJECT_ID, timestamp=int(0e9), x=static_cartesian_state[C_X], y=static_cartesian_state[C_Y],
                   z=DEFAULT_OBJECT_Z_VALUE,
                   yaw=static_cartesian_state[C_YAW], size=car_size, confidence=0, v_x=static_cartesian_state[C_V],
                   v_y=0,
                   acceleration_lon=0, omega_yaw=0, steering_angle=0)

@pytest.fixture(scope='function')
def init_state(init_ego_state, init_dyn_obj) -> State:
    yield State(ego_state=init_ego_state, dynamic_objects=[init_dyn_obj], occupancy_state=OccupancyState(0, np.array([]), np.array([])))


@pytest.fixture(scope='function')
def predicted_dyn_object_states_constant_yaw(predicted_cartesian_state_0: CartesianState,
                                             predicted_cartesian_state_1: CartesianState,
                                             predicted_cartesian_state_2_constant_yaw: CartesianState,
                                             prediction_timestamps: np.ndarray) -> List[DynamicObject]:
    object_states = [
        DynamicObject(obj_id=DYNAMIC_OBJECT_ID, timestamp=int(prediction_timestamps[0] * 1e9), x=predicted_cartesian_state_0[C_X],
                      y=predicted_cartesian_state_0[C_Y],
                      z=DEFAULT_OBJECT_Z_VALUE,
                      yaw=predicted_cartesian_state_0[C_YAW], size=car_size, confidence=0,
                      v_x=predicted_cartesian_state_0[C_V],
                      v_y=0,
                      acceleration_lon=0, omega_yaw=0),
        DynamicObject(obj_id=DYNAMIC_OBJECT_ID, timestamp=int(prediction_timestamps[1] * 1e9), x=predicted_cartesian_state_1[C_X],
                      y=predicted_cartesian_state_1[C_Y],
                      z=DEFAULT_OBJECT_Z_VALUE,
                      yaw=predicted_cartesian_state_1[C_YAW], size=car_size, confidence=0,
                      v_x=predicted_cartesian_state_1[C_V],
                      v_y=0,
                      acceleration_lon=0, omega_yaw=0),
        DynamicObject(obj_id=DYNAMIC_OBJECT_ID, timestamp=int(prediction_timestamps[2] * 1e9),
                      x=predicted_cartesian_state_2_constant_yaw[C_X], y=predicted_cartesian_state_2_constant_yaw[C_Y],
                      z=DEFAULT_OBJECT_Z_VALUE,
                      yaw=predicted_cartesian_state_2_constant_yaw[C_YAW], size=car_size, confidence=0,
                      v_x=predicted_cartesian_state_2_constant_yaw[C_V],
                      v_y=0,
                      acceleration_lon=0, omega_yaw=0)]
    yield object_states


@pytest.fixture(scope='function')
def predicted_dyn_object_states_road_yaw(predicted_cartesian_state_0: CartesianState,
                                         predicted_cartesian_state_1: CartesianState,
                                         predicted_cartesian_state_2_road_yaw: CartesianState,
                                         prediction_timestamps: np.ndarray)-> List[DynamicObject]:
    object_states = [
        DynamicObject(obj_id=DYNAMIC_OBJECT_ID, timestamp=int(prediction_timestamps[0] * 1e9), x=predicted_cartesian_state_0[C_X],
                      y=predicted_cartesian_state_0[C_Y],
                      z=DEFAULT_OBJECT_Z_VALUE,
                      yaw=predicted_cartesian_state_0[C_YAW], size=car_size, confidence=0,
                      v_x=predicted_cartesian_state_0[C_V],
                      v_y=0,
                      acceleration_lon=0, omega_yaw=0),
        DynamicObject(obj_id=DYNAMIC_OBJECT_ID, timestamp=int(prediction_timestamps[1] * 1e9), x=predicted_cartesian_state_1[C_X],
                      y=predicted_cartesian_state_1[C_Y],
                      z=DEFAULT_OBJECT_Z_VALUE,
                      yaw=predicted_cartesian_state_1[C_YAW], size=car_size, confidence=0,
                      v_x=predicted_cartesian_state_1[C_V],
                      v_y=0,
                      acceleration_lon=0, omega_yaw=0),
        DynamicObject(obj_id=DYNAMIC_OBJECT_ID, timestamp=int(prediction_timestamps[2] * 1e9),
                      x=predicted_cartesian_state_2_road_yaw[C_X], y=predicted_cartesian_state_2_road_yaw[C_Y],
                      z=DEFAULT_OBJECT_Z_VALUE,
                      yaw=predicted_cartesian_state_2_road_yaw[C_YAW], size=car_size, confidence=0,
                      v_x=predicted_cartesian_state_2_road_yaw[C_V],
                      v_y=0,
                      acceleration_lon=0, omega_yaw=0)]
    yield object_states


@pytest.fixture(scope='function')
def predicted_static_ego_states(static_cartesian_state: CartesianState, prediction_timestamps: np.ndarray) -> List[EgoState]:
    ego_states = [EgoState(obj_id=EGO_OBJECT_ID, timestamp=int(prediction_timestamps[0] * 1e9), x=static_cartesian_state[C_X],
                           y=static_cartesian_state[C_Y],
                           z=DEFAULT_OBJECT_Z_VALUE,
                           yaw=static_cartesian_state[C_YAW], size=car_size, confidence=0,
                           v_x=static_cartesian_state[C_V],
                           v_y=0,
                           acceleration_lon=0, omega_yaw=0, steering_angle=0),
                  EgoState(obj_id=EGO_OBJECT_ID, timestamp=int(prediction_timestamps[1] * 1e9), x=static_cartesian_state[C_X],
                           y=static_cartesian_state[C_Y],
                           z=DEFAULT_OBJECT_Z_VALUE,
                           yaw=static_cartesian_state[C_YAW], size=car_size, confidence=0,
                           v_x=static_cartesian_state[C_V],
                           v_y=0,
                           acceleration_lon=0, omega_yaw=0, steering_angle=0),
                  EgoState(obj_id=EGO_OBJECT_ID, timestamp=int(prediction_timestamps[2] * 1e9), x=static_cartesian_state[C_X],
                           y=static_cartesian_state[C_Y],
                           z=DEFAULT_OBJECT_Z_VALUE,
                           yaw=static_cartesian_state[C_YAW], size=car_size, confidence=0,
                           v_x=static_cartesian_state[C_V],
                           v_y=0,
                           acceleration_lon=0, omega_yaw=0, steering_angle=0)]
    yield ego_states
