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
from decision_making.src.planning.behavioral.data_objects import DynamicActionRecipe, StaticActionRecipe
from decision_making.test.planning.behavioral.behavioral_state_fixtures import all_follow_lane_recipes
from rte.python.logger.AV_logger import AV_Logger

import numpy as np

from decision_making.test.planning.behavioral.behavioral_state_fixtures import state_without_objects_ovalmilford, MILFORD_NAVIGATION_PLAN
from decision_making.test.messages.static_scene_fixture import scene_static_ovalmilford



def test_filter_FollowLaneFilterActionsWithTooHighLateralAcceleration_FilteredAccordingly(
        scene_static_ovalmilford,
        state_without_objects_ovalmilford,
        all_follow_lane_recipes: List[StaticActionRecipe]):

    logger = AV_Logger.get_logger()
    SceneStaticModel.get_instance().set_scene_static(scene_static_ovalmilford)

    behavioral_state = BehavioralGridState.create_from_state(state_without_objects_ovalmilford, MILFORD_NAVIGATION_PLAN, logger)

    static_action_space = StaticActionSpace(logger, filtering=DEFAULT_STATIC_RECIPE_FILTERING)
    action_specs = static_action_space.specify_goals(all_follow_lane_recipes, behavioral_state)

    filtering = ActionSpecFiltering(filters=[FilterIfNone(),
                                             FilterForKinematics(),
                                             FilterByLateralAcceleration('predicates')], logger=logger)

    action_specs = action_specs[46:47]
    mask = filtering.filter_action_specs(action_specs, behavioral_state)
    expected_mask = [False if spec is None else spec.v < 30 for spec in action_specs]
    expected_mask = expected_mask[46:47]
    assert mask == expected_mask
