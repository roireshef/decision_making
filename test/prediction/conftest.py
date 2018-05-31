from typing import List

import numpy as np
import pytest

from decision_making.src.global_constants import DEFAULT_OBJECT_Z_VALUE
from decision_making.src.planning.trajectory.trajectory_planner import SamplableTrajectory
from decision_making.src.planning.types import CartesianState, C_X, C_Y, C_YAW, C_V
from decision_making.src.prediction.action_unaware_prediction.physical_time_alignment_predictor import \
    PhysicalTimeAlignmentPredictor
from decision_making.src.prediction.ego_aware_prediction.maneuver_based_predictor import ManeuverBasedPredictor
from decision_making.src.prediction.ego_aware_prediction.maneuver_recognition.constant_velocity_maneuver_classifier import \
    ConstantVelocityManeuverClassifier

from decision_making.src.prediction.ego_aware_prediction.trajectory_generation.werling_trajectory_generator import \
    WerlingTrajectoryGenerator
from decision_making.src.state.state import DynamicObject, ObjectSize, EgoState, State, OccupancyState
from decision_making.test.planning.trajectory.mock_samplable_trajectory import MockSamplableTrajectory
from rte.python.logger.AV_logger import AV_Logger

DYNAMIC_OBJECT_ID = 1
EGO_OBJECT_ID = 0

PREDICTION_HORIZON = 6

@pytest.fixture(scope='function')
def physical_time_alignment_predictor() -> PhysicalTimeAlignmentPredictor:
    logger = AV_Logger.get_logger("PREDICTOR_TEST_LOGGER")
    yield PhysicalTimeAlignmentPredictor(logger)

@pytest.fixture(scope='function')
def werling_trajectory_generator() -> WerlingTrajectoryGenerator:
    yield WerlingTrajectoryGenerator()

@pytest.fixture(scope='function')
def constant_velocity_predictor(werling_trajectory_generator: WerlingTrajectoryGenerator) -> ManeuverBasedPredictor:
    logger = AV_Logger.get_logger("PREDICTOR_TEST_LOGGER")
    maneuver_classifier = ConstantVelocityManeuverClassifier()
    predictor = ManeuverBasedPredictor(logger, maneuver_classifier=maneuver_classifier,
                                       trajectory_generator=werling_trajectory_generator)
    yield predictor

@pytest.fixture(scope='function')
def car_size() -> ObjectSize:
    yield ObjectSize(length=3.0, width=2.0, height=1.2)


@pytest.fixture(scope='function')
def init_cartesian_state() -> CartesianState:
    yield np.array([500.0, 0.0, 0.0, 10.0])


@pytest.fixture(scope='function')
def predicted_cartesian_state_0() -> CartesianState:
    yield np.array([510.0, 0.0, 0.0, 10.0])


@pytest.fixture(scope='function')
def predicted_cartesian_state_1_constant_yaw() -> CartesianState:
    yield np.array([590.0, 0.0, 0.0, 10.0])


@pytest.fixture(scope='function')
def predicted_cartesian_state_1_road_yaw() -> CartesianState:
    yield np.array([590.0, 0.0, 0.0, 10.0])


@pytest.fixture(scope='function')
def predicted_cartesian_state_2_constant_yaw() -> CartesianState:
    yield np.array([700.0, 0.0, 0.0, 10.0])


@pytest.fixture(scope='function')
def predicted_cartesian_state_2_road_yaw() -> CartesianState:
    yield np.array([700.0, 0.0, 0.0, 10.0])


@pytest.fixture(scope='function')
def static_cartesian_state() -> CartesianState:
    yield np.array([50.0, 0.0, 0.0, 0.0])


@pytest.fixture(scope='function')
def prediction_timestamps() -> np.array:
    yield np.array([1.0, 9.0, 20.0])


@pytest.fixture(scope='function')
def init_dyn_obj(init_cartesian_state) -> DynamicObject:
    yield DynamicObject(obj_id=DYNAMIC_OBJECT_ID, timestamp=int(0e9), x=init_cartesian_state[C_X],
                        y=init_cartesian_state[C_Y],
                        z=DEFAULT_OBJECT_Z_VALUE,
                        yaw=init_cartesian_state[C_YAW], size=car_size, confidence=0, v_x=init_cartesian_state[C_V],
                        v_y=0,
                        acceleration_lon=0, curvature=0)


@pytest.fixture(scope='function')
def init_ego_state(static_cartesian_state) -> EgoState:
    yield EgoState(obj_id=EGO_OBJECT_ID, timestamp=int(0e9), x=static_cartesian_state[C_X],
                   y=static_cartesian_state[C_Y],
                   z=DEFAULT_OBJECT_Z_VALUE,
                   yaw=static_cartesian_state[C_YAW], size=car_size, confidence=0, v_x=static_cartesian_state[C_V],
                   v_y=0,
                   acceleration_lon=0, curvature=0)


@pytest.fixture(scope='function')
def init_state(init_ego_state, init_dyn_obj) -> State:
    yield State(ego_state=init_ego_state, dynamic_objects=[init_dyn_obj],
                occupancy_state=OccupancyState(0, np.array([]), np.array([])))


@pytest.fixture(scope='function')
def unaligned_dynamic_object(predicted_cartesian_state_1_constant_yaw, prediction_timestamps):
    yield DynamicObject(obj_id=DYNAMIC_OBJECT_ID, timestamp=int(prediction_timestamps[1] * 1e9),
                                   x=predicted_cartesian_state_1_constant_yaw[C_X],
                                   y=predicted_cartesian_state_1_constant_yaw[C_Y],
                                   z=DEFAULT_OBJECT_Z_VALUE,
                                   yaw=predicted_cartesian_state_1_constant_yaw[C_YAW], size=car_size, confidence=0,
                                   v_x=predicted_cartesian_state_1_constant_yaw[C_V],
                                   v_y=0,
                                   acceleration_lon=0, curvature=0)


@pytest.fixture(scope='function')
def aligned_ego_state(init_ego_state, unaligned_dynamic_object):
    # Changing only timestamp since ego's speed is 0
    init_ego_state.timestamp = unaligned_dynamic_object.timestamp
    yield init_ego_state


@pytest.fixture(scope='function')
def unaligned_state(init_ego_state, unaligned_dynamic_object) -> State:
    yield State(ego_state=init_ego_state, dynamic_objects=[unaligned_dynamic_object],
                occupancy_state=OccupancyState(0, np.array([]), np.array([])))


@pytest.fixture(scope='function')
def predicted_dyn_object_states_constant_yaw(predicted_cartesian_state_0: CartesianState,
                                             predicted_cartesian_state_1_constant_yaw: CartesianState,
                                             predicted_cartesian_state_2_constant_yaw: CartesianState,
                                             prediction_timestamps: np.ndarray) -> List[DynamicObject]:
    object_states = [
        DynamicObject(obj_id=DYNAMIC_OBJECT_ID, timestamp=int(prediction_timestamps[0] * 1e9),
                      x=predicted_cartesian_state_0[C_X],
                      y=predicted_cartesian_state_0[C_Y],
                      z=DEFAULT_OBJECT_Z_VALUE,
                      yaw=predicted_cartesian_state_0[C_YAW], size=car_size, confidence=0,
                      v_x=predicted_cartesian_state_0[C_V],
                      v_y=0,
                      acceleration_lon=0, curvature=0),
        DynamicObject(obj_id=DYNAMIC_OBJECT_ID, timestamp=int(prediction_timestamps[1] * 1e9), x=predicted_cartesian_state_1_constant_yaw[C_X],
                      y=predicted_cartesian_state_1_constant_yaw[C_Y],
                      z=DEFAULT_OBJECT_Z_VALUE,
                      yaw=predicted_cartesian_state_1_constant_yaw[C_YAW], size=car_size, confidence=0,
                      v_x=predicted_cartesian_state_1_constant_yaw[C_V],
                      v_y=0,
                      acceleration_lon=0, curvature=0),
        DynamicObject(obj_id=DYNAMIC_OBJECT_ID, timestamp=int(prediction_timestamps[2] * 1e9),
                      x=predicted_cartesian_state_2_constant_yaw[C_X], y=predicted_cartesian_state_2_constant_yaw[C_Y],
                      z=DEFAULT_OBJECT_Z_VALUE,
                      yaw=predicted_cartesian_state_2_constant_yaw[C_YAW], size=car_size, confidence=0,
                      v_x=predicted_cartesian_state_2_constant_yaw[C_V],
                      v_y=0,
                      acceleration_lon=0, curvature=0)]
    yield object_states


@pytest.fixture(scope='function')
def predicted_dyn_object_states_road_yaw(predicted_cartesian_state_0: CartesianState,
                                         predicted_cartesian_state_1_road_yaw: CartesianState,
                                         predicted_cartesian_state_2_road_yaw: CartesianState,
                                         prediction_timestamps: np.ndarray) -> List[DynamicObject]:
    object_states = [
        DynamicObject(obj_id=DYNAMIC_OBJECT_ID, timestamp=int(prediction_timestamps[0] * 1e9),
                      x=predicted_cartesian_state_0[C_X],
                      y=predicted_cartesian_state_0[C_Y],
                      z=DEFAULT_OBJECT_Z_VALUE,
                      yaw=predicted_cartesian_state_0[C_YAW], size=car_size, confidence=0,
                      v_x=predicted_cartesian_state_0[C_V],
                      v_y=0,
                      acceleration_lon=0, curvature=0),
        DynamicObject(obj_id=DYNAMIC_OBJECT_ID, timestamp=int(prediction_timestamps[1] * 1e9), x=predicted_cartesian_state_1_road_yaw[C_X],
                      y=predicted_cartesian_state_1_road_yaw[C_Y],
                      z=DEFAULT_OBJECT_Z_VALUE,
                      yaw=predicted_cartesian_state_1_road_yaw[C_YAW], size=car_size, confidence=0,
                      v_x=predicted_cartesian_state_1_road_yaw[C_V],
                      v_y=0,
                      acceleration_lon=0, curvature=0),
        DynamicObject(obj_id=DYNAMIC_OBJECT_ID, timestamp=int(prediction_timestamps[2] * 1e9),
                      x=predicted_cartesian_state_2_road_yaw[C_X], y=predicted_cartesian_state_2_road_yaw[C_Y],
                      z=DEFAULT_OBJECT_Z_VALUE,
                      yaw=predicted_cartesian_state_2_road_yaw[C_YAW], size=car_size, confidence=0,
                      v_x=predicted_cartesian_state_2_road_yaw[C_V],
                      v_y=0,
                      acceleration_lon=0, curvature=0)]
    yield object_states


@pytest.fixture(scope='function')
def predicted_static_ego_states(static_cartesian_state: CartesianState, prediction_timestamps: np.ndarray) -> List[
    EgoState]:
    ego_states = [
        EgoState(obj_id=EGO_OBJECT_ID, timestamp=int(prediction_timestamps[0] * 1e9), x=static_cartesian_state[C_X],
                 y=static_cartesian_state[C_Y],
                 z=DEFAULT_OBJECT_Z_VALUE,
                 yaw=static_cartesian_state[C_YAW], size=car_size, confidence=0,
                 v_x=static_cartesian_state[C_V],
                 v_y=0,
                 acceleration_lon=0, curvature=0),
        EgoState(obj_id=EGO_OBJECT_ID, timestamp=int(prediction_timestamps[1] * 1e9), x=static_cartesian_state[C_X],
                 y=static_cartesian_state[C_Y],
                 z=DEFAULT_OBJECT_Z_VALUE,
                 yaw=static_cartesian_state[C_YAW], size=car_size, confidence=0,
                 v_x=static_cartesian_state[C_V],
                 v_y=0,
                 acceleration_lon=0, curvature=0),
        EgoState(obj_id=EGO_OBJECT_ID, timestamp=int(prediction_timestamps[2] * 1e9), x=static_cartesian_state[C_X],
                 y=static_cartesian_state[C_Y],
                 z=DEFAULT_OBJECT_Z_VALUE,
                 yaw=static_cartesian_state[C_YAW], size=car_size, confidence=0,
                 v_x=static_cartesian_state[C_V],
                 v_y=0,
                 acceleration_lon=0, curvature=0)]
def predicted_static_ego_states(static_cartesian_state: CartesianState, prediction_timestamps: np.ndarray) -> List[EgoState]:
    ego_states = [EgoState(obj_id=EGO_OBJECT_ID, timestamp=int(prediction_timestamps[0] * 1e9), x=static_cartesian_state[C_X],
                           y=static_cartesian_state[C_Y],
                           z=DEFAULT_OBJECT_Z_VALUE,
                           yaw=static_cartesian_state[C_YAW], size=car_size, confidence=0,
                           v_x=static_cartesian_state[C_V],
                           v_y=0,
                           acceleration_lon=0, curvature=0),
                  EgoState(obj_id=EGO_OBJECT_ID, timestamp=int(prediction_timestamps[1] * 1e9), x=static_cartesian_state[C_X],
                           y=static_cartesian_state[C_Y],
                           z=DEFAULT_OBJECT_Z_VALUE,
                           yaw=static_cartesian_state[C_YAW], size=car_size, confidence=0,
                           v_x=static_cartesian_state[C_V],
                           v_y=0,
                           acceleration_lon=0, curvature=0),
                  EgoState(obj_id=EGO_OBJECT_ID, timestamp=int(prediction_timestamps[2] * 1e9), x=static_cartesian_state[C_X],
                           y=static_cartesian_state[C_Y],
                           z=DEFAULT_OBJECT_Z_VALUE,
                           yaw=static_cartesian_state[C_YAW], size=car_size, confidence=0,
                           v_x=static_cartesian_state[C_V],
                           v_y=0,
                           acceleration_lon=0, curvature=0)]
    yield ego_states


@pytest.fixture(scope='function')
def ego_samplable_trajectory(static_cartesian_state) -> SamplableTrajectory:
    a_k_zero_array = np.array([0, 0])
    cartesian_extended_trajectory = np.array(
        [np.append(static_cartesian_state, a_k_zero_array),
         np.append(static_cartesian_state, a_k_zero_array),
         np.append(static_cartesian_state, a_k_zero_array)])
    yield MockSamplableTrajectory(cartesian_extended_trajectory)