import numpy as np
import pytest
from unittest.mock import patch

from decision_making.src.infra.pubsub import PubSub
from decision_making.src.global_constants import VELOCITY_MINIMAL_THRESHOLD, BEHAVIORAL_PLANNING_NAME_FOR_LOGGING
from decision_making.src.messages.scene_dynamic_message import SceneDynamic
from decision_making.src.planning.types import FS_SV
from decision_making.src.state.state import DynamicObjectsData, State
from rte.python.logger.AV_logger import AV_Logger
from decision_making.test.planning.custom_fixtures import dynamic_objects_not_on_road, \
    scene_dynamic_fix_single_host_hypothesis, scene_dynamic_fix_two_host_hypotheses, \
    scene_dynamic_fix_three_host_hypotheses, pubsub, dynamic_objects_negative_velocity
from decision_making.test.messages.scene_static_fixture import scene_static_oval_with_splits, scene_static_pg_split
from decision_making.test.planning.custom_fixtures import turn_signal


# @patch(FILTER_OBJECT_OFF_ROAD_PATH, False)
def test_dynamicObjCallbackWithoutFilter_objectOffRoad_stateWithObject(pubsub: PubSub,
                                                                       dynamic_objects_not_on_road: DynamicObjectsData):
    """
    :param pubsub: Inter-process communication interface.
    :param scene_dynamic_fix: Fixture of scene dynamic

    Checking functionality of dynamic_object_callback for an object that is not on the road.
    """
    # Inserting a object that's not on the road
    dyn_obj_list = State.create_dyn_obj_list(dynamic_objects_not_on_road)
    assert len(dyn_obj_list) == 1  # check that object was inserted


def test_createStateFromSceneDyamic_singleHostHypothesis_correctHostLocalization(pubsub: PubSub,
                                                                                 scene_dynamic_fix_single_host_hypothesis: SceneDynamic,
                                                                                 turn_signal):
    """
    :param scene_dynamic_fix: Fixture of scene dynamic
    :param gff_segment_ids: GFF lane segment ids for last action

    Checking functionality of create_state_from_scene_dynamic for the case of single host hypothesis
    """

    logger = AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING)

    gff_segment_ids = np.array([200, 210, 220])

    state = State.create_state_from_scene_dynamic(scene_dynamic=scene_dynamic_fix_single_host_hypothesis,
                                                  selected_gff_segment_ids=gff_segment_ids,
                                                  logger=logger, turn_signal=turn_signal)

    assert state.ego_state.map_state.lane_id == 200


def test_createStateFromSceneDyamic_twoHostHypotheses_correctHostLocalization(pubsub: PubSub,
                                                                              scene_dynamic_fix_two_host_hypotheses: SceneDynamic,
                                                                              turn_signal):
    """
    :param scene_dynamic_fix: Fixture of scene dynamic
    :param gff_segment_ids: GFF lane segment ids for last action

    Checking functionality of create_state_from_scene_dynamic for the case of multiple host hypothesis
    """

    logger = AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING)

    gff_segment_ids = np.array([201, 211, 221])

    state = State.create_state_from_scene_dynamic(scene_dynamic=scene_dynamic_fix_two_host_hypotheses,
                                                  selected_gff_segment_ids=gff_segment_ids,
                                                  logger=logger, turn_signal=turn_signal)

    assert state.ego_state.map_state.lane_id == 201


def test_createStateFromSceneDyamic_threeHostHypotheses_correctHostLocalization(pubsub: PubSub,
                                                                                scene_dynamic_fix_three_host_hypotheses: SceneDynamic,
                                                                                turn_signal):
    """
    :param scene_dynamic_fix: Fixture of scene dynamic
    :param gff_segment_ids: GFF lane segment ids for last action

    Checking functionality of create_state_from_scene_dynamic for the case of multiple host hypothesis
    """

    logger = AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING)

    gff_segment_ids = np.array([2244100, 19670533, 58375685])

    state = State.create_state_from_scene_dynamic(scene_dynamic=scene_dynamic_fix_three_host_hypotheses,
                                                  selected_gff_segment_ids=gff_segment_ids,
                                                  logger=logger, turn_signal=turn_signal)

    assert state.ego_state.map_state.lane_id == 2244100


def test_createStateFromSceneDyamic_noGFFDifferentEndCosts_correctHostLocalization(pubsub: PubSub,
                                                                                   scene_dynamic_fix_three_host_hypotheses: SceneDynamic,
                                                                                   turn_signal):
    """
    :param scene_dynamic_fix: Fixture of scene dynamic
    :param gff_segment_ids: GFF lane segment ids for last action

    Checking functionality of create_state_from_scene_dynamic for the case of multiple host hypothesis
    """

    logger = AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING)

    gff_segment_ids = np.array([])
    route_plan_dict = {2244100: (0, 1), 19670532: (0, 0), 19670533: (0, 1)}

    state = State.create_state_from_scene_dynamic(scene_dynamic=scene_dynamic_fix_three_host_hypotheses,
                                                  selected_gff_segment_ids=gff_segment_ids,
                                                  route_plan_dict=route_plan_dict,
                                                  logger=logger, turn_signal=turn_signal)

    assert state.ego_state.map_state.lane_id == 19670532


def test_createStateFromSceneDyamic_noGFFSimilarEndCosts_correctHostLocalization(pubsub: PubSub,
                                                                                 scene_dynamic_fix_three_host_hypotheses: SceneDynamic,
                                                                                 turn_signal):
    """
    :param scene_dynamic_fix: Fixture of scene dynamic
    :param gff_segment_ids: GFF lane segment ids for last action

    Checking functionality of create_state_from_scene_dynamic for the case of multiple host hypothesis
    """

    logger = AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING)

    gff_segment_ids = np.array([])
    route_plan_dict = {2244100: (0, 1), 19670532: (0, 0), 19670533: (0, 0)}

    state = State.create_state_from_scene_dynamic(scene_dynamic=scene_dynamic_fix_three_host_hypotheses,
                                                  selected_gff_segment_ids=gff_segment_ids,
                                                  route_plan_dict=route_plan_dict,
                                                  logger=logger, turn_signal=turn_signal)

    assert state.ego_state.map_state.lane_id == 19670532


@pytest.mark.skip(reason="irrelevant since was moved to SP")
def test_dynamicObjCallback_negativeVelocity_stateWithUpdatedVelocity(dynamic_objects_negative_velocity: DynamicObjectsData):
    """
    :param pubsub: Inter-process communication interface.
    :param scene_dynamic_fix: Fixture of scene dynamic

    Checking functionality of dynamic_object_callback for an object that is not on the road.
    """

    dyn_obj_list = State.create_dyn_obj_list(dynamic_objects_negative_velocity)

    assert len(dyn_obj_list) == 1 # check that object was inserted
    assert np.isclose(dyn_obj_list[0].map_state.lane_fstate[FS_SV], VELOCITY_MINIMAL_THRESHOLD)


@pytest.mark.skip(reason="irrelevant since was moved to SP")
def test_dynamicObjCallbackWithFilter_objectOffRoad_stateWithoutObject(dynamic_objects_not_on_road: DynamicObjectsData):
    """
    :param pubsub: Inter-process communication interface.
    :param scene_dynamic_fix: Fixture of scene dynamic

    Checking functionality of dynamic_object_callback for an object that is not on the road.
    """

    # Inserting a object that's not on the road
    dyn_obj_list = State.create_dyn_obj_list(dynamic_objects_not_on_road)
    assert len(dyn_obj_list) == 0   # check that object was not inserted
