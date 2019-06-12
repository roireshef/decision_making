from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.utils.map_utils import MapUtils
from decision_making.test.messages.scene_static_fixture import scene_static_testable
from typing import List

import numpy as np
import pytest

from decision_making.src.planning.trajectory.samplable_trajectory import SamplableTrajectory
from decision_making.src.planning.types import CartesianExtendedState, C_YAW
from decision_making.src.prediction.action_unaware_prediction.physical_time_alignment_predictor import \
    PhysicalTimeAlignmentPredictor
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.prediction.ego_aware_prediction.maneuver_based_predictor import ManeuverBasedPredictor
from decision_making.src.prediction.ego_aware_prediction.maneuver_recognition.constant_velocity_maneuver_classifier import \
    ConstantVelocityManeuverClassifier
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor

from decision_making.src.prediction.ego_aware_prediction.trajectory_generation.werling_trajectory_generator import \
    WerlingTrajectoryGenerator
from decision_making.src.state.state import DynamicObject, ObjectSize, EgoState, State, OccupancyState
from decision_making.test.planning.trajectory.mock_samplable_trajectory import MockSamplableTrajectory
from rte.python.logger.AV_logger import AV_Logger

DYNAMIC_OBJECT_ID = 1
EGO_OBJECT_ID = 0

PREDICTION_HORIZON = 6

CARTESIAN_CREATION = 'decision_making.src.state.state.DynamicObject.create_from_cartesian_state'


@pytest.fixture(scope='function')
def physical_time_alignment_predictor() -> PhysicalTimeAlignmentPredictor:
    logger = AV_Logger.get_logger("PREDICTOR_TEST_LOGGER")
    yield PhysicalTimeAlignmentPredictor(logger)


@pytest.fixture(scope='function')
def werling_trajectory_generator() -> WerlingTrajectoryGenerator:
    yield WerlingTrajectoryGenerator()


@pytest.fixture(scope='function')
def constant_velocity_predictor(werling_trajectory_generator: WerlingTrajectoryGenerator) -> EgoAwarePredictor:
    logger = AV_Logger.get_logger("PREDICTOR_TEST_LOGGER")
    maneuver_classifier = ConstantVelocityManeuverClassifier()
    predictor = ManeuverBasedPredictor(logger, maneuver_classifier=maneuver_classifier,
                                       trajectory_generator=werling_trajectory_generator)
    yield predictor

@pytest.fixture(scope='function')
def road_following_predictor() -> EgoAwarePredictor:
    logger = AV_Logger.get_logger("PREDICTOR_TEST_LOGGER")
    predictor = RoadFollowingPredictor(logger)
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
def init_ego_cartesian_state() -> CartesianExtendedState:
    yield np.array([50.0, 0.0, 0.0, 1.0, 0.0, 0.0])


@pytest.fixture(scope='function')
def predicted_ego_cartesian_state_0() -> CartesianExtendedState:
    yield np.array([51.0, 0.0, 0.0, 1.0, 0.0, 0.0])


@pytest.fixture(scope='function')
def predicted_ego_cartesian_state_1() -> CartesianExtendedState:
    yield np.array([59.0, 0.0, 0.0, 1.0, 0.0, 0.0])

@pytest.fixture(scope='function')
def predicted_ego_cartesian_state_2() -> CartesianExtendedState:
    yield np.array([70.0, 0.0, 0.0, 1.0, 0.0, 0.0])

@pytest.fixture(scope='function')
def prediction_timestamps() -> np.array:
    yield np.array([1.0, 9.0, 20.0])


@pytest.fixture(scope='function')
def init_dyn_obj(init_cartesian_state: CartesianExtendedState, car_size: ObjectSize) -> DynamicObject:
    yield DynamicObject.create_from_cartesian_state(obj_id=DYNAMIC_OBJECT_ID, timestamp=int(0e9),
                                                    cartesian_state=init_cartesian_state, size=car_size,
                                                    confidence=0, off_map=False)


@pytest.fixture(scope='function')
def init_ego_state(static_cartesian_state: CartesianExtendedState, car_size: ObjectSize) -> EgoState:
    yield EgoState.create_from_cartesian_state(obj_id=EGO_OBJECT_ID, timestamp=int(0e9),
                                               cartesian_state=static_cartesian_state, size=car_size, confidence=0, off_map=False)

@pytest.fixture(scope='function')
def init_dynamic_ego_state(init_ego_cartesian_state: CartesianExtendedState, car_size: ObjectSize) -> EgoState:
    yield EgoState.create_from_cartesian_state(obj_id=EGO_OBJECT_ID, timestamp=int(0e9),
                                               cartesian_state=init_ego_cartesian_state, size=car_size, confidence=0, off_map=False)


@pytest.fixture(scope='function')
def init_state(init_ego_state: EgoState, init_dyn_obj: DynamicObject) -> State:
    yield State(is_sampled=False, ego_state=init_ego_state, dynamic_objects=[init_dyn_obj],
                occupancy_state=OccupancyState(0, np.array([]), np.array([])))


@pytest.fixture(scope='function')
def dynamic_init_state(init_dynamic_ego_state: EgoState, init_dyn_obj: DynamicObject) -> State:
    yield State(is_sampled=False, ego_state=init_dynamic_ego_state, dynamic_objects=[init_dyn_obj],
                occupancy_state=OccupancyState(0, np.array([]), np.array([])))


@pytest.fixture(scope='function')
def unaligned_dynamic_object(predicted_cartesian_state_1_constant_yaw: CartesianExtendedState, prediction_timestamps,
                             car_size):
    yield DynamicObject.create_from_cartesian_state(obj_id=DYNAMIC_OBJECT_ID,
                                                    timestamp=int(prediction_timestamps[1] * 1e9),
                                                    cartesian_state=predicted_cartesian_state_1_constant_yaw,
                                                    size=car_size, confidence=0, off_map=False)


@pytest.fixture(scope='function')
def aligned_ego_state(init_ego_state, unaligned_dynamic_object):
    # Changing only timestamp since ego's speed is 0
    init_ego_state.timestamp = unaligned_dynamic_object.timestamp
    yield init_ego_state


@pytest.fixture(scope='function')
def unaligned_state(init_ego_state, unaligned_dynamic_object) -> State:
    yield State(is_sampled=False, ego_state=init_ego_state, dynamic_objects=[unaligned_dynamic_object],
                occupancy_state=OccupancyState(0, np.array([]), np.array([])))


@pytest.fixture(scope='function')
def predicted_dyn_object_states_constant_yaw(predicted_cartesian_state_0: CartesianExtendedState,
                                             predicted_cartesian_state_1_constant_yaw: CartesianExtendedState,
                                             predicted_cartesian_state_2_constant_yaw: CartesianExtendedState,
                                             prediction_timestamps: np.ndarray, car_size: ObjectSize) -> List[
    DynamicObject]:
    object_states = [
        DynamicObject.create_from_cartesian_state(obj_id=DYNAMIC_OBJECT_ID,
                                                  timestamp=int(prediction_timestamps[0] * 1e9),
                                                  cartesian_state=predicted_cartesian_state_0, size=car_size,
                                                  confidence=0, off_map=False),
        DynamicObject.create_from_cartesian_state(obj_id=DYNAMIC_OBJECT_ID,
                                                  timestamp=int(prediction_timestamps[1] * 1e9),
                                                  cartesian_state=predicted_cartesian_state_1_constant_yaw,
                                                  size=car_size, confidence=0, off_map=False),

        DynamicObject.create_from_cartesian_state(obj_id=DYNAMIC_OBJECT_ID,
                                                  timestamp=int(prediction_timestamps[2] * 1e9),
                                                  cartesian_state=predicted_cartesian_state_2_constant_yaw,
                                                  size=car_size, confidence=0, off_map=False)
    ]

    yield object_states


@pytest.fixture(scope='function')
def predicted_dyn_object_states_road_yaw(predicted_cartesian_state_0: CartesianExtendedState,
                                         predicted_cartesian_state_1_road_yaw: CartesianExtendedState,
                                         predicted_cartesian_state_2_road_yaw: CartesianExtendedState,
                                         prediction_timestamps: np.ndarray) -> List[DynamicObject]:
    object_states = [
        DynamicObject.create_from_cartesian_state(obj_id=DYNAMIC_OBJECT_ID,
                                                  timestamp=int(prediction_timestamps[0] * 1e9),
                                                  cartesian_state=predicted_cartesian_state_0, size=car_size,
                                                  confidence=0, off_map=False),
        DynamicObject.create_from_cartesian_state(obj_id=DYNAMIC_OBJECT_ID,
                                                  timestamp=int(prediction_timestamps[1] * 1e9),
                                                  cartesian_state=predicted_cartesian_state_1_road_yaw,
                                                  size=car_size, confidence=0, off_map=False),

        DynamicObject.create_from_cartesian_state(obj_id=DYNAMIC_OBJECT_ID,
                                                  timestamp=int(prediction_timestamps[2] * 1e9),
                                                  cartesian_state=predicted_cartesian_state_2_road_yaw,
                                                  size=car_size, confidence=0, off_map=False)
    ]

    yield object_states


@pytest.fixture(scope='function')
def predicted_static_ego_states(static_cartesian_state: CartesianExtendedState, prediction_timestamps: np.ndarray) -> \
        List[EgoState]:
    ego_states = [
        EgoState.create_from_cartesian_state(obj_id=EGO_OBJECT_ID,
                                             timestamp=int(prediction_timestamps[0] * 1e9),
                                             cartesian_state=static_cartesian_state, size=car_size,
                                             confidence=0, off_map=False),
        EgoState.create_from_cartesian_state(obj_id=EGO_OBJECT_ID,
                                             timestamp=int(prediction_timestamps[1] * 1e9),
                                             cartesian_state=static_cartesian_state,
                                             size=car_size, confidence=0, off_map=False),

        EgoState.create_from_cartesian_state(obj_id=EGO_OBJECT_ID,
                                             timestamp=int(prediction_timestamps[2] * 1e9),
                                             cartesian_state=static_cartesian_state,
                                             size=car_size, confidence=0, off_map=False)
    ]

    yield ego_states


@pytest.fixture(scope='function')
def predicted_dynamic_ego_states(predicted_ego_cartesian_state_0: CartesianExtendedState,
                                 predicted_ego_cartesian_state_1: CartesianExtendedState,
                                 predicted_ego_cartesian_state_2: CartesianExtendedState,
                                         prediction_timestamps: np.ndarray) -> List[EgoState]:
    ego_states = [
        EgoState.create_from_cartesian_state(obj_id=EGO_OBJECT_ID,
                                             timestamp=int(prediction_timestamps[0] * 1e9),
                                             cartesian_state=predicted_ego_cartesian_state_0, size=car_size,
                                             confidence=0, off_map=False),
        EgoState.create_from_cartesian_state(obj_id=EGO_OBJECT_ID,
                                             timestamp=int(prediction_timestamps[1] * 1e9),
                                             cartesian_state=predicted_ego_cartesian_state_1,
                                             size=car_size, confidence=0, off_map=False),

        EgoState.create_from_cartesian_state(obj_id=EGO_OBJECT_ID,
                                             timestamp=int(prediction_timestamps[2] * 1e9),
                                             cartesian_state=predicted_ego_cartesian_state_2,
                                             size=car_size, confidence=0, off_map=False)
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
def original_state_with_sorrounding_objects(scene_static_testable):
    SceneStaticModel.get_instance().set_scene_static(scene_static_testable)

    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    # Ego state
    road_segment_id = 1
    ego_lane_lon = 15.0
    ego_lane_ordinal = 1

    lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_id)[ego_lane_ordinal]
    frenet_frame = MapUtils.get_lane_frenet_frame(lane_id)
    cstate = frenet_frame.fstate_to_cstate(np.array([ego_lane_lon, 0, 0, 0, 0, 0]))
    ego_pos = cstate[:C_YAW]
    ego_yaw = cstate[C_YAW]

    ego_state = EgoState.create_from_cartesian_state(obj_id=0, timestamp=0, cartesian_state=np.array(
        [ego_pos[0], ego_pos[1], ego_yaw, 0.0, 0.0, 0.0]),
                                                     size=car_size, confidence=1.0, off_map=False)

    # Generate objects at the following locations:
    obj_id = 1
    obj_lane_lons = [5.0, 10.0, 15.0, 20.0, 25.0]
    obj_lane_ordinals = [0, 1, 2]

    dynamic_objects: List[DynamicObject] = list()
    for obj_lane_lon in obj_lane_lons:
        for obj_lane_ordinal in obj_lane_ordinals:

            if obj_lane_lon == ego_lane_lon and obj_lane_ordinal == ego_lane_ordinal:
                # Don't create an object where the ego is
                continue

            lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_id)[obj_lane_ordinal]
            frenet_frame = MapUtils.get_lane_frenet_frame(lane_id)
            cstate = frenet_frame.fstate_to_cstate(np.array([obj_lane_lon, 0, 0, 0, 0, 0]))
            obj_pos = cstate[:C_YAW]
            obj_yaw = cstate[C_YAW]

            dynamic_object = DynamicObject.create_from_cartesian_state(obj_id=obj_id, timestamp=0,
                                                                       cartesian_state=np.array(
                                                                              [obj_pos[0], obj_pos[1], obj_yaw, 0.0,
                                                                               0.0, 0.0]),
                                                                       size=car_size, confidence=1.0, off_map=False)

            dynamic_objects.append(dynamic_object)
            obj_id += 1

    yield State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)
