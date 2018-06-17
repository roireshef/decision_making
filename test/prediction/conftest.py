from typing import List

import numpy as np
import pytest

from decision_making.src.global_constants import DEFAULT_OBJECT_Z_VALUE
from decision_making.src.planning.trajectory.trajectory_planner import SamplableTrajectory
from decision_making.src.planning.types import CartesianExtendedState, C_X, C_Y, C_YAW, C_V
from decision_making.src.prediction.action_unaware_prediction.physical_time_alignment_predictor import \
    PhysicalTimeAlignmentPredictor
from decision_making.src.prediction.ego_aware_prediction.maneuver_based_predictor import ManeuverBasedPredictor
from decision_making.src.prediction.ego_aware_prediction.maneuver_recognition.constant_velocity_maneuver_classifier import \
    ConstantVelocityManeuverClassifier

from decision_making.src.prediction.ego_aware_prediction.trajectory_generation.werling_trajectory_generator import \
    WerlingTrajectoryGenerator
from decision_making.src.state.state import NewDynamicObject, ObjectSize, NewEgoState, State, OccupancyState
from decision_making.test.planning.trajectory.mock_samplable_trajectory import MockSamplableTrajectory
from mapping.src.service.map_service import MapService
from mapping.test.model.testable_map_fixtures import map_api_mock
from rte.python.logger.AV_logger import AV_Logger
from unittest.mock import patch

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
def init_cartesian_state() -> CartesianExtendedState:
    yield np.array([500.0, 0.0, 0.0, 10.0, 0.0, 0.0])


@pytest.fixture(scope='function')
def predicted_cartesian_state_0() -> CartesianExtendedState:
    yield np.array([510.0, 0.0, 0.0, 10.0, 0.0, 0.0])


@pytest.fixture(scope='function')
def predicted_cartesian_state_1_constant_yaw() -> CartesianExtendedState:
    yield np.array([590.0, 0.0, 0.0, 10.0, 0.0, 0.0])


@pytest.fixture(scope='function')
def predicted_cartesian_state_1_road_yaw() -> CartesianExtendedState:
    yield np.array([590.0, 0.0, 0.0, 10.0, 0.0, 0.0])


@pytest.fixture(scope='function')
def predicted_cartesian_state_2_constant_yaw() -> CartesianExtendedState:
    yield np.array([700.0, 0.0, 0.0, 10.0, 0.0, 0.0])


@pytest.fixture(scope='function')
def predicted_cartesian_state_2_road_yaw() -> CartesianExtendedState:
    yield np.array([700.0, 0.0, 0.0, 10.0, 0.0, 0.0])


@pytest.fixture(scope='function')
def static_cartesian_state() -> CartesianExtendedState:
    yield np.array([50.0, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.fixture(scope='function')
def prediction_timestamps() -> np.array:
    yield np.array([1.0, 9.0, 20.0])


@pytest.fixture(scope='function')
def init_dyn_obj(init_cartesian_state: CartesianExtendedState, car_size: ObjectSize) -> NewDynamicObject:
    yield NewDynamicObject.create_from_cartesian_state(obj_id=DYNAMIC_OBJECT_ID, timestamp=int(0e9),
                                                       cartesian_state=init_cartesian_state, size=car_size,
                                                       confidence=0)


@pytest.fixture(scope='function')
def init_ego_state(static_cartesian_state: CartesianExtendedState, car_size: ObjectSize) -> NewEgoState:
    yield NewEgoState.create_from_cartesian_state(obj_id=EGO_OBJECT_ID, timestamp=int(0e9),
                                                  cartesian_state=static_cartesian_state, size=car_size, confidence=0)


@pytest.fixture(scope='function')
def init_state(init_ego_state: NewEgoState, init_dyn_obj: NewDynamicObject) -> State:
    yield State(ego_state=init_ego_state, dynamic_objects=[init_dyn_obj],
                occupancy_state=OccupancyState(0, np.array([]), np.array([])))


@pytest.fixture(scope='function')
def unaligned_dynamic_object(predicted_cartesian_state_1_constant_yaw: CartesianExtendedState, prediction_timestamps,
                             car_size):
    yield NewDynamicObject.create_from_cartesian_state(obj_id=DYNAMIC_OBJECT_ID,
                                                       timestamp=int(prediction_timestamps[1] * 1e9),
                                                       cartesian_state=predicted_cartesian_state_1_constant_yaw,
                                                       size=car_size, confidence=0)


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
def predicted_dyn_object_states_constant_yaw(predicted_cartesian_state_0: CartesianExtendedState,
                                             predicted_cartesian_state_1_constant_yaw: CartesianExtendedState,
                                             predicted_cartesian_state_2_constant_yaw: CartesianExtendedState,
                                             prediction_timestamps: np.ndarray, car_size: ObjectSize) -> List[
    NewDynamicObject]:
    object_states = [
        NewDynamicObject.create_from_cartesian_state(obj_id=DYNAMIC_OBJECT_ID,
                                                     timestamp=int(prediction_timestamps[0] * 1e9),
                                                     cartesian_state=predicted_cartesian_state_0, size=car_size,
                                                     confidence=0),
        NewDynamicObject.create_from_cartesian_state(obj_id=DYNAMIC_OBJECT_ID,
                                                     timestamp=int(prediction_timestamps[1] * 1e9),
                                                     cartesian_state=predicted_cartesian_state_1_constant_yaw,
                                                     size=car_size, confidence=0),

        NewDynamicObject.create_from_cartesian_state(obj_id=DYNAMIC_OBJECT_ID,
                                                     timestamp=int(prediction_timestamps[2] * 1e9),
                                                     cartesian_state=predicted_cartesian_state_2_constant_yaw,
                                                     size=car_size, confidence=0)
    ]

    yield object_states


@pytest.fixture(scope='function')
def predicted_dyn_object_states_road_yaw(predicted_cartesian_state_0: CartesianExtendedState,
                                         predicted_cartesian_state_1_road_yaw: CartesianExtendedState,
                                         predicted_cartesian_state_2_road_yaw: CartesianExtendedState,
                                         prediction_timestamps: np.ndarray) -> List[NewDynamicObject]:
    object_states = [
        NewDynamicObject.create_from_cartesian_state(obj_id=DYNAMIC_OBJECT_ID,
                                                     timestamp=int(prediction_timestamps[0] * 1e9),
                                                     cartesian_state=predicted_cartesian_state_0, size=car_size,
                                                     confidence=0),
        NewDynamicObject.create_from_cartesian_state(obj_id=DYNAMIC_OBJECT_ID,
                                                     timestamp=int(prediction_timestamps[1] * 1e9),
                                                     cartesian_state=predicted_cartesian_state_1_road_yaw,
                                                     size=car_size, confidence=0),

        NewDynamicObject.create_from_cartesian_state(obj_id=DYNAMIC_OBJECT_ID,
                                                     timestamp=int(prediction_timestamps[2] * 1e9),
                                                     cartesian_state=predicted_cartesian_state_2_road_yaw,
                                                     size=car_size, confidence=0)
    ]

    yield object_states


@pytest.fixture(scope='function')
def predicted_static_ego_states(static_cartesian_state: CartesianExtendedState, prediction_timestamps: np.ndarray) -> \
        List[NewEgoState]:
    ego_states = [
        NewEgoState.create_from_cartesian_state(obj_id=EGO_OBJECT_ID,
                                                timestamp=int(prediction_timestamps[0] * 1e9),
                                                cartesian_state=static_cartesian_state, size=car_size,
                                                confidence=0),
        NewEgoState.create_from_cartesian_state(obj_id=EGO_OBJECT_ID,
                                                timestamp=int(prediction_timestamps[1] * 1e9),
                                                cartesian_state=static_cartesian_state,
                                                size=car_size, confidence=0),

        NewEgoState.create_from_cartesian_state(obj_id=EGO_OBJECT_ID,
                                                timestamp=int(prediction_timestamps[2] * 1e9),
                                                cartesian_state=static_cartesian_state,
                                                size=car_size, confidence=0)
    ]

    yield ego_states


@pytest.fixture(scope='function')
def ego_samplable_trajectory(static_cartesian_state: CartesianExtendedState) -> SamplableTrajectory:
    cartesian_extended_trajectory = np.array(
        [static_cartesian_state,
         static_cartesian_state,
         static_cartesian_state])
    yield MockSamplableTrajectory(cartesian_extended_trajectory)


@pytest.fixture(scope='function')
def original_state_with_sorrounding_objects():
    with patch.object(MapService, 'get_instance', map_api_mock):
        # Stub of occupancy grid
        occupancy_state = OccupancyState(0, np.array([]), np.array([]))

        car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

        # Ego state
        ego_road_id = 1
        ego_road_lon = 15.0
        ego_road_lat = 4.5

        ego_pos, ego_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id=ego_road_id,
                                                                                        lon=ego_road_lon,
                                                                                        lat=ego_road_lat)

        ego_state = NewEgoState.create_from_cartesian_state(obj_id=0, timestamp=0, cartesian_state=np.array(
            [ego_pos[0], ego_pos[1], ego_yaw, 0.0, 0.0, 0.0]),
                                                            size=car_size, confidence=1.0)

        # Generate objects at the following locations:
        obj_id = 1
        obj_road_id = 1
        obj_road_lons = [5.0, 10.0, 15.0, 20.0, 25.0]
        obj_road_lats = [1.5, 4.5, 6.0]

        dynamic_objects: List[NewDynamicObject] = list()
        for obj_road_lon in obj_road_lons:
            for obj_road_lat in obj_road_lats:

                if obj_road_lon == ego_road_lon and obj_road_lat == ego_road_lat:
                    # Don't create an object where the ego is
                    continue

                obj_pos, obj_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id=obj_road_id,
                                                                                                lon=obj_road_lon,
                                                                                                lat=obj_road_lat)

                dynamic_object = NewDynamicObject.create_from_cartesian_state(obj_id=obj_id, timestamp=0,
                                                                              cartesian_state=np.array(
                                                                                  [obj_pos[0], obj_pos[1], obj_yaw, 0.0,
                                                                                   0.0, 0.0]),
                                                                              size=car_size, confidence=1.0)

                dynamic_objects.append(dynamic_object)
                obj_id += 1

        yield State(occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)
