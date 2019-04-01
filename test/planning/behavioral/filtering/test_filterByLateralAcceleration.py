from decision_making.src.planning.behavioral.default_config import DEFAULT_DYNAMIC_RECIPE_FILTERING, \
    DEFAULT_STATIC_RECIPE_FILTERING
from decision_making.src.planning.behavioral.filtering.action_spec_filter_bank import FilterIfNone, \
    FilterByLateralAcceleration, FilterForKinematics
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import ActionSpecFiltering
from decision_making.src.scene.scene_static_model import SceneStaticModel
from logging import Logger
from typing import List

import time

from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import DynamicActionRecipe, StaticActionRecipe, ActionSpec, \
    RelativeLane, AggressivenessLevel
from decision_making.test.planning.behavioral.behavioral_state_fixtures import all_follow_lane_recipes
from rte.python.logger.AV_logger import AV_Logger

import numpy as np

from decision_making.test.planning.behavioral.behavioral_state_fixtures import state_without_objects_ovalmilford, state_before_curvature_ovalmilford,\
    MILFORD_NAVIGATION_PLAN
from decision_making.test.messages.static_scene_fixture import scene_static_ovalmilford


# TODO: create a better test
def test_filter_FollowLaneFilterActionsWithTooHighLateralAcceleration_FilteredAccordingly(
        scene_static_ovalmilford,
        state_before_curvature_ovalmilford,
        all_follow_lane_recipes: List[StaticActionRecipe]):

    logger = AV_Logger.get_logger()
    SceneStaticModel.get_instance().set_scene_static(scene_static_ovalmilford)

    behavioral_state = BehavioralGridState.create_from_state(state_before_curvature_ovalmilford, MILFORD_NAVIGATION_PLAN, logger)

    # High Curvature is located around s=300
    action_specs = [ActionSpec(10, 10, 110, 0, StaticActionRecipe(RelativeLane.SAME_LANE, 10, AggressivenessLevel.CALM)),
                    ActionSpec(10, 34, 300, 0, StaticActionRecipe(RelativeLane.SAME_LANE, 34, AggressivenessLevel.CALM))]

    filtering = ActionSpecFiltering(filters=[FilterIfNone(),
                                             FilterByLateralAcceleration('predicates')], logger=logger)
    mask = filtering.filter_action_specs(action_specs, behavioral_state)
    expected_mask = [True, False]
    assert mask == expected_mask
