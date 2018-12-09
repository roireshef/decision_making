import numpy as np
from decision_making.src.dm_main import NAVIGATION_PLAN, DEFAULT_MAP_FILE
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpaceContainer
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import RelativeLane
from decision_making.src.planning.behavioral.default_config import DEFAULT_DYNAMIC_RECIPE_FILTERING, \
    DEFAULT_STATIC_RECIPE_FILTERING
from decision_making.src.planning.behavioral.evaluators.rule_based_action_spec_evaluator import \
    RuleBasedActionSpecEvaluator
from decision_making.src.planning.behavioral.evaluators.zero_value_approximator import ZeroValueApproximator
from decision_making.src.planning.behavioral.filtering.action_spec_filter_bank import FilterIfNone
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import ActionSpecFiltering
from decision_making.src.planning.behavioral.planner.single_step_behavioral_planner import SingleStepBehavioralPlanner
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.state.state import EgoState, State, ObjectSize
from decision_making.src.utils.map_utils import MapUtils
from decision_making.test.messages.static_scene_fixture import scene_static
from mapping.src.service.map_service import MapService
from rte.python.logger.AV_logger import AV_Logger
from decision_making.test.planning.behavioral.behavioral_state_fixtures import NAVIGATION_PLAN


def test_generateTerminalStates_multiRoad_accurate(scene_static):

    SceneStaticModel.get_instance().set_scene_static(scene_static)

    logger = AV_Logger.get_logger()
    predictor = RoadFollowingPredictor(logger)
    nav_plan = NAVIGATION_PLAN
    MapService.initialize('PG_split.bin')
    size = ObjectSize(4, 2, 1)
    timestamp_in_sec = 1000
    timestamp = int(timestamp_in_sec * 1e9)

    ego_cstate = MapUtils.get_lane_frenet_frame(201).fstate_to_cstate(np.array([60, 14, 0, 0, 0, 0]))
    ego = EgoState.create_from_cartesian_state(obj_id=0, timestamp=timestamp, cartesian_state=ego_cstate, size=size,
                                               confidence=0)

    obj1_cstate = MapUtils.get_lane_frenet_frame(201).fstate_to_cstate(np.array([80, 14, 0, 0, 0, 0]))
    obj1 = EgoState.create_from_cartesian_state(obj_id=1, timestamp=timestamp, cartesian_state=obj1_cstate,
                                                size=size, confidence=0)

    obj2_cstate = MapUtils.get_lane_frenet_frame(200).fstate_to_cstate(np.array([70, 10, 0, 0, 0, 0]))
    obj2 = EgoState.create_from_cartesian_state(obj_id=2, timestamp=timestamp, cartesian_state=obj2_cstate,
                                                size=size, confidence=0)

    obj3_cstate = MapUtils.get_lane_frenet_frame(202).fstate_to_cstate(np.array([90, 16, 0, 0, 0, 0]))
    obj3 = EgoState.create_from_cartesian_state(obj_id=3, timestamp=timestamp, cartesian_state=obj3_cstate,
                                                size=size, confidence=0)

    obj4_cstate = MapUtils.get_lane_frenet_frame(211).fstate_to_cstate(np.array([5, 17, 0, 0, 0, 0]))
    obj4 = EgoState.create_from_cartesian_state(obj_id=4, timestamp=timestamp, cartesian_state=obj4_cstate,
                                                size=size, confidence=0)

    obj5_cstate = MapUtils.get_lane_frenet_frame(212).fstate_to_cstate(np.array([10, 15, 0, 0, 0, 0]))
    obj5 = EgoState.create_from_cartesian_state(obj_id=5, timestamp=timestamp, cartesian_state=obj5_cstate,
                                                size=size, confidence=0)

    obj6_cstate = MapUtils.get_lane_frenet_frame(200).fstate_to_cstate(np.array([30, 15, 0, 0, 0, 0]))
    obj6 = EgoState.create_from_cartesian_state(obj_id=6, timestamp=timestamp, cartesian_state=obj6_cstate,
                                                size=size, confidence=0)

    state = State(None, [obj1, obj2, obj3, obj4, obj5, obj6], ego)

    action_space = ActionSpaceContainer(logger, [StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING),
                                                 DynamicActionSpace(logger, predictor,
                                                                    DEFAULT_DYNAMIC_RECIPE_FILTERING)])
    recipe_evaluator = None
    action_spec_evaluator = RuleBasedActionSpecEvaluator(logger)
    value_approximator = ZeroValueApproximator(logger)
    action_spec_filtering = ActionSpecFiltering(filters=[FilterIfNone()], logger=logger)

    planner = SingleStepBehavioralPlanner(action_space, recipe_evaluator, action_spec_evaluator,
                                          action_spec_filtering,
                                          value_approximator, predictor, logger)

    behavioral_state = BehavioralGridState.create_from_state(state, nav_plan, logger)

    recipes_mask = action_space.filter_recipes(action_space.recipes, behavioral_state)
    valid_action_recipes = [action_recipe for i, action_recipe in enumerate(action_space.recipes) if
                            recipes_mask[i]]

    action_specs = np.full(len(action_space.recipes), None)
    action_specs[recipes_mask] = action_space.specify_goals(valid_action_recipes, behavioral_state)
    action_specs = list(action_specs)
    action_specs_mask = action_spec_filtering.filter_action_specs(action_specs, behavioral_state)

    terminal_behavioral_states = planner._generate_terminal_states(state, behavioral_state, action_specs,
                                                                   action_specs_mask, nav_plan)

    assert len(terminal_behavioral_states) == len(action_specs)
    assert len([state for state in terminal_behavioral_states if state is not None]) == sum(action_specs_mask)
    assert np.isclose(
        np.array([state.ego_state.timestamp_in_sec for state in terminal_behavioral_states if state is not None]),
        np.array([timestamp_in_sec + spec.t for i, spec in enumerate(action_specs) if action_specs_mask[i]])).all()
    assert (np.array(
        [len(state.road_occupancy_grid) for state in terminal_behavioral_states if state is not None]) > 0).all()

    for i, state in enumerate(terminal_behavioral_states):
        if state is not None:
            for cell, objects_with_semantics in state.road_occupancy_grid.items():
                if abs(action_specs[i].relative_lane.value + cell[0].value) <= 1:
                    originial_rel_lane = RelativeLane(action_specs[i].relative_lane.value + cell[0].value)
                    for obj in objects_with_semantics:
                        relevant_idxs = extended_lane_frame.has_segment_ids(object_segment_ids)
                        for rel_lane, extended_lane_frame in extended_lane_frames.items():  # loop over at most 3 unified frames
                            # find all targets belonging to the current unified frame
                            relevant_idxs = extended_lane_frame.has_segment_ids(object_segment_ids)
                            if relevant_idxs.any():
                                # convert relevant dynamic objects to fstate w.r.t. the current unified frame
                                object_extended_fstates[relevant_idxs] = extended_lane_frame.convert_from_segment_states(
                                    object_segment_fstates[relevant_idxs], object_segment_ids[relevant_idxs])
