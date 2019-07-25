from unittest.mock import patch

import numpy as np
import pytest

from decision_making.src.infra.pubsub import PubSub
from decision_making.src.global_constants import STATE_MODULE_NAME_FOR_LOGGING, VELOCITY_MINIMAL_THRESHOLD
from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.messages.scene_dynamic_message import SceneDynamic
from decision_making.src.planning.types import FS_SV
from decision_making.src.state.state_module import StateModule, DynamicObjectsData
from rte.python.logger.AV_logger import AV_Logger
from decision_making.test.planning.custom_fixtures import dynamic_objects_not_on_road, scene_dynamic_fix, pubsub, \
    dynamic_objects_negative_velocity
from decision_making.test.messages.scene_static_fixture import scene_static_pg_split, scene_static_testable


# @patch(FILTER_OBJECT_OFF_ROAD_PATH, False)
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

# def test_createStateFromSceneDyamic_singleHostHypothesis_correctHostLocalization(pubsub: PubSub,
#                                                                                  scene_dynamic_multiple_host_hypothesis: SceneDynamic,
#                                                                                  gff_segment_ids: np.ndarray):
#     """
#     :param scene_dynamic_fix: Fixture of scene dynamic
#     :param gff_segment_ids: GFF lane segment ids for last action
#
#     Checking functionality of create_state_from_scene_dynamic for the case of multiple host hypothesis
#     """
#
#     logger = AV_Logger.get_logger(STATE_MODULE_NAME_FOR_LOGGING)
#
#     state_module = StateModule(pubsub=pubsub, logger=logger, scene_dynamic=scene_dynamic_multiple_host_hypothesis)
#     state_module.start()
#
#     state = state_module.create_state_from_scene_dynamic(scene_dynamic_multiple_host_hypothesis, gff_segment_ids)
#
#     assert state.ego_state.map_state.lane_id == 1


@pytest.mark.skip(reason="irrelevant since was moved to SP")
def test_dynamicObjCallback_negativeVelocity_stateWithUpdatedVelocity(pubsub: PubSub,
                                                                      dynamic_objects_negative_velocity: DynamicObjectsData,
                                                                      scene_dynamic_fix: SceneDynamic,
                                                                      scene_static_pg_split):
    """
    :param pubsub: Inter-process communication interface.
    :param scene_dynamic_fix: Fixture of scene dynamic

    Checking functionality of dynamic_object_callback for an object that is not on the road.
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)

    logger = AV_Logger.get_logger(STATE_MODULE_NAME_FOR_LOGGING)

    state_module = StateModule(pubsub=pubsub, logger=logger,
                               scene_dynamic=scene_dynamic_fix)
    state_module.start()

    dyn_obj_list = state_module.create_dyn_obj_list(dynamic_objects_negative_velocity)

    assert len(dyn_obj_list) == 1 # check that object was inserted
    assert np.isclose(dyn_obj_list[0].map_state.lane_fstate[FS_SV], VELOCITY_MINIMAL_THRESHOLD)


@pytest.mark.skip(reason="irrelevant since was moved to SP")
def test_dynamicObjCallbackWithFilter_objectOffRoad_stateWithoutObject(pubsub: PubSub,
                                                                       dynamic_objects_not_on_road: DynamicObjectsData,
                                                                       scene_dynamic_fix: SceneDynamic,
                                                                       scene_static_pg_split):
    """
    :param pubsub: Inter-process communication interface.
    :param scene_dynamic_fix: Fixture of scene dynamic

    Checking functionality of dynamic_object_callback for an object that is not on the road.
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)

    logger = AV_Logger.get_logger(STATE_MODULE_NAME_FOR_LOGGING)

    state_module = StateModule(pubsub=pubsub, logger=logger,
                               scene_dynamic=scene_dynamic_fix)
    state_module.start()
    # Inserting a object that's not on the road
    dyn_obj_list = state_module.create_dyn_obj_list(dynamic_objects_not_on_road)
    assert len(dyn_obj_list) == 0   # check that object was not inserted
