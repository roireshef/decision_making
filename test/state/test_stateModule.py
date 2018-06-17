from unittest.mock import patch

from common_data.src.communication.pubsub.pubsub import PubSub
from decision_making.src.global_constants import STATE_MODULE_NAME_FOR_LOGGING
from decision_making.src.state.state import OccupancyState, NewEgoState
from decision_making.src.state.state_module import StateModule
from decision_making.test.constants import MAP_SERVICE_ABSOLUTE_PATH, FILTER_OBJECT_OFF_ROAD_PATH
from gm_lcm import LcmPerceivedDynamicObjectList
from rte.python.logger.AV_logger import AV_Logger
from decision_making.test.planning.custom_fixtures import dynamic_objects_not_in_fov, dynamic_objects_in_fov,\
    dynamic_objects_not_on_road, ego_state_fix, pubsub
from mapping.test.model.testable_map_fixtures import map_api_mock
import numpy as np

@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_dynamicObjCallback_objectInAndOutOfFOV_stateWithInFOVObject(pubsub: PubSub,
                                                                     dynamic_objects_in_fov: LcmPerceivedDynamicObjectList,
                                                                     dynamic_objects_not_in_fov: LcmPerceivedDynamicObjectList,
                                                                     ego_state_fix: NewEgoState):
    """
    :param pubsub: Inter-process communication interface.
    :param dynamic_objects_in_fov: Fixture of a serialized dynamic object data located within the field of view
            (FOV).
    :param dynamic_objects_not_in_fov: Fixture of a serialized dynamic object with the same id as above but now it's located
                                        out of the field of view.
    :param ego_state_fix: Fixture of an ego state compatible with the above two fixtures.

    This test checks the memory functionality of the StateModule. It initially sends into the StateModule a dynamic object
    in the FOV followed by a message indicating that the object is out of FOV. The test then asserts that the last known
    object properties have been "remembered".
    """
    logger = AV_Logger.get_logger(STATE_MODULE_NAME_FOR_LOGGING)

    state_module = StateModule(pubsub=pubsub, logger=logger,
                               occupancy_state=OccupancyState(0, np.array([]), np.array([])),
                               dynamic_objects=None, ego_state=ego_state_fix)
    state_module.start()
    #Inserting a object in_fov in order to remember it.
    state_module.create_dyn_obj_list(dynamic_objects_in_fov)
    new_dyn_obj_list = state_module.create_dyn_obj_list(dynamic_objects_not_in_fov)
    assert len(new_dyn_obj_list) == 1
    assert new_dyn_obj_list[0].timestamp == dynamic_objects_in_fov.timestamp

    # the object should be loaded from memory and this is why its location and speed remains the same as in in_fov fixture
    assert new_dyn_obj_list[0].x == dynamic_objects_in_fov.dynamic_objects[0].location.x
    assert new_dyn_obj_list[0].y == dynamic_objects_in_fov.dynamic_objects[0].location.y
    glob_v_x = dynamic_objects_in_fov.dynamic_objects[0].velocity.v_x
    glob_v_y = dynamic_objects_in_fov.dynamic_objects[0].velocity.v_y
    yaw = dynamic_objects_in_fov.dynamic_objects[0].bbox.yaw

    # convert velocity from map coordinates to relative to its own yaw
    v_x = np.cos(yaw) * glob_v_x + np.sin(yaw) * glob_v_y
    v_y = -np.sin(yaw) * glob_v_x + np.cos(yaw) * glob_v_y

    assert new_dyn_obj_list[0].v_x == v_x
    assert new_dyn_obj_list[0].v_y == v_y

    state_module.stop()


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_isObjectOnRoad_objectOffOfRoad_False(pubsub: PubSub, ego_state_fix: NewEgoState):
    """
    :param pubsub: Inter-process communication interface.
    :param ego_state_fix: Fixture of an ego state.

    Checking functionality of _is_object_on_road for an object that is off the road.
    """
    logger = AV_Logger.get_logger(STATE_MODULE_NAME_FOR_LOGGING)

    state_module = StateModule(pubsub=pubsub, logger=logger,
                               occupancy_state=OccupancyState(0, np.array([]), np.array([])),
                               dynamic_objects=None, ego_state=ego_state_fix)
    actual_result = state_module._is_object_on_road(np.array([17.0,17.0,0.0]), 0.0)
    expected_result = False
    assert expected_result == actual_result

@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_isObjectOnRoad_objectOnRoad_True(pubsub: PubSub, ego_state_fix: NewEgoState):
    """
    :param pubsub: Inter-process communication interface.
    :param ego_state_fix: Fixture of an ego state.

    Checking functionality of _is_object_on_road for an object that is on the road.
    """
    logger = AV_Logger.get_logger(STATE_MODULE_NAME_FOR_LOGGING)

    state_module = StateModule(pubsub=pubsub, logger=logger,
                               occupancy_state=OccupancyState(0, np.array([]), np.array([])),
                               dynamic_objects=None, ego_state=ego_state_fix)
    actual_result = state_module._is_object_on_road(np.array([5.0,1.0,0.0]), 0.0)
    expected_result = True
    assert expected_result == actual_result


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
@patch(FILTER_OBJECT_OFF_ROAD_PATH, False)
def test_dynamicObjCallbackWithoutFilter_objectOffRoad_stateWithObject(pubsub: PubSub,
                                                                                   dynamic_objects_not_on_road: LcmPerceivedDynamicObjectList,
                                                                                   ego_state_fix: NewEgoState):
    """
    :param pubsub: Inter-process communication interface.
    :param ego_state_fix: Fixture of an ego state.

    Checking functionality of dynamic_object_callback for an object that is not on the road.
    """
    logger = AV_Logger.get_logger(STATE_MODULE_NAME_FOR_LOGGING)

    state_module = StateModule(pubsub=pubsub, logger=logger,
                               occupancy_state=OccupancyState(0, np.array([]), np.array([])),
                               dynamic_objects=None, ego_state=ego_state_fix)
    state_module.start()
    # Inserting a object that's not on the road
    dyn_obj_list = state_module.create_dyn_obj_list(dynamic_objects_not_on_road)
    assert len(dyn_obj_list) == 1 # check that object was inserted


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_dynamicObjCallbackWithFilter_objectOffRoad_stateWithoutObject(pubsub: PubSub,
                                                                                   dynamic_objects_not_on_road: LcmPerceivedDynamicObjectList,
                                                                                   ego_state_fix: NewEgoState):
    """
    :param pubsub: Inter-process communication interface.
    :param ego_state_fix: Fixture of an ego state.

    Checking functionality of dynamic_object_callback for an object that is not on the road.
    """
    logger = AV_Logger.get_logger(STATE_MODULE_NAME_FOR_LOGGING)

    state_module = StateModule(pubsub=pubsub, logger=logger,
                               occupancy_state=OccupancyState(0, np.array([]), np.array([])),
                               dynamic_objects=None, ego_state=ego_state_fix)
    state_module.start()
    # Inserting a object that's not on the road
    dyn_obj_list = state_module.create_dyn_obj_list(dynamic_objects_not_on_road)
    assert len(dyn_obj_list) == 0   # check that object was not inserted