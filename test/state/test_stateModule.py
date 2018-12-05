from unittest.mock import patch

import numpy as np
import pytest

from common_data.interface.py.idl_generated_files.Rte_Types import LcmPerceivedDynamicObjectList
from common_data.src.communication.pubsub.pubsub import PubSub
from decision_making.src.global_constants import STATE_MODULE_NAME_FOR_LOGGING, VELOCITY_MINIMAL_THRESHOLD
from decision_making.src.mapping.scene_model import SceneModel
from decision_making.src.messages.scene_dynamic_message import SceneDynamic
from decision_making.src.messages.scene_static_message import SceneStatic
from decision_making.src.planning.types import FS_SV
from decision_making.src.state.state_module import StateModule, DynamicObjectsData
from decision_making.test.constants import MAP_SERVICE_ABSOLUTE_PATH, FILTER_OBJECT_OFF_ROAD_PATH
from mapping.test.model.testable_map_fixtures import map_api_mock
from rte.python.logger.AV_logger import AV_Logger
from decision_making.test.planning.custom_fixtures import dynamic_objects_not_on_road, scene_dynamic_fix, pubsub, \
    dynamic_objects_negative_velocity

from decision_making.test.messages.static_scene_fixture import scene_static

@pytest.mark.skip(reason="Irrelevent when no out-of-fov data is available")
@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_dynamicObjCallback_objectInAndOutOfFOV_stateWithInFOVObject(pubsub: PubSub,
                                                                     dynamic_objects_in_fov: LcmPerceivedDynamicObjectList,
                                                                     dynamic_objects_not_in_fov: LcmPerceivedDynamicObjectList,
                                                                     scene_dynamic_fix: SceneDynamic):
    """
    :param pubsub: Inter-process communication interface.
    :param dynamic_objects_in_fov: Fixture of a serialized dynamic object data located within the field of view
            (FOV).
    :param dynamic_objects_not_in_fov: Fixture of a serialized dynamic object with the same id as above but now it's located
                                        out of the field of view.
    :param scene_dynamic_fix: Fixture of scene dynamic compatible with the above two fixtures.

    This test checks the memory functionality of the StateModule. It initially sends into the StateModule a dynamic object
    in the FOV followed by a message indicating that the object is out of FOV. The test then asserts that the last known
    object properties have been "remembered".
    """
    logger = AV_Logger.get_logger(STATE_MODULE_NAME_FOR_LOGGING)

    state_module = StateModule(pubsub=pubsub, logger=logger,
                               scene_dynamic=scene_dynamic_fix)
    state_module.start()
    # Inserting an object in_fov in order to remember it.
    state_module.create_dyn_obj_list(dynamic_objects_in_fov)
    new_dyn_obj_list = state_module.create_dyn_obj_list(dynamic_objects_not_in_fov)
    assert len(new_dyn_obj_list) == 1
    assert new_dyn_obj_list[0].timestamp == dynamic_objects_in_fov.timestamp

    # The object should be loaded from memory and this is why its location and speed remains the
    #  same as in in_fov fixture
    assert new_dyn_obj_list[0].x == dynamic_objects_in_fov.dynamic_objects[0].location.x
    assert new_dyn_obj_list[0].y == dynamic_objects_in_fov.dynamic_objects[0].location.y

    velocity = new_dyn_obj_list[0].velocity

    yaw = dynamic_objects_in_fov.dynamic_objects[0].bbox.yaw

    v_x = velocity * np.cos(yaw)
    v_y = velocity * np.sin(yaw)

    assert np.isclose(dynamic_objects_in_fov.dynamic_objects[0].velocity.v_x , v_x, atol=1e-3)
    assert np.isclose(dynamic_objects_in_fov.dynamic_objects[0].velocity.v_y, v_y, atol=1e-3)

    state_module.stop()


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
@patch(FILTER_OBJECT_OFF_ROAD_PATH, False)
def test_dynamicObjCallbackWithoutFilter_objectOffRoad_stateWithObject(pubsub: PubSub,
                                                                       dynamic_objects_not_on_road: DynamicObjectsData,
                                                                       scene_dynamic_fix: SceneDynamic):
    """
    :param pubsub: Inter-process communication interface.
    :param scene_dynamic_fix: Fixture of scene dynamic

    Checking functionality of dynamic_object_callback for an object that is not on the road.
    """
    logger = AV_Logger.get_logger(STATE_MODULE_NAME_FOR_LOGGING)

    state_module = StateModule(pubsub=pubsub, logger=logger, scene_dynamic=scene_dynamic_fix)
    state_module.start()
    # Inserting a object that's not on the road
    dyn_obj_list = state_module.create_dyn_obj_list(dynamic_objects_not_on_road)
    assert len(dyn_obj_list) == 1  # check that object was inserted


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_dynamicObjCallback_negativeVelocity_stateWithUpdatedVelocity(pubsub: PubSub,
                                                                      dynamic_objects_negative_velocity: DynamicObjectsData,
                                                                      scene_dynamic_fix: SceneDynamic):
    """
    :param pubsub: Inter-process communication interface.
    :param scene_dynamic_fix: Fixture of scene dynamic

    Checking functionality of dynamic_object_callback for an object that is not on the road.
    """
    logger = AV_Logger.get_logger(STATE_MODULE_NAME_FOR_LOGGING)

    state_module = StateModule(pubsub=pubsub, logger=logger,
                               scene_dynamic=scene_dynamic_fix)
    state_module.start()

    dyn_obj_list = state_module.create_dyn_obj_list(dynamic_objects_negative_velocity)

    assert len(dyn_obj_list) == 1 # check that object was inserted
    assert np.isclose(dyn_obj_list[0].map_state.lane_fstate[FS_SV], VELOCITY_MINIMAL_THRESHOLD)


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_dynamicObjCallbackWithFilter_objectOffRoad_stateWithoutObject(pubsub: PubSub,
                                                                       dynamic_objects_not_on_road: DynamicObjectsData,
                                                                       scene_dynamic_fix: SceneDynamic,
                                                                       scene_static: SceneStatic):
    """
    :param pubsub: Inter-process communication interface.
    :param scene_dynamic_fix: Fixture of scene dynamic

    Checking functionality of dynamic_object_callback for an object that is not on the road.
    """

    SceneModel.get_instance().set_scene_static(scene_static)
    logger = AV_Logger.get_logger(STATE_MODULE_NAME_FOR_LOGGING)

    state_module = StateModule(pubsub=pubsub, logger=logger,
                               scene_dynamic=scene_dynamic_fix)
    state_module.start()
    # Inserting a object that's not on the road
    dyn_obj_list = state_module.create_dyn_obj_list(dynamic_objects_not_on_road)
    assert len(dyn_obj_list) == 0   # check that object was not inserted