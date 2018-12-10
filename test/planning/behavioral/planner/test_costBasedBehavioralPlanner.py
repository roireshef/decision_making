import numpy as np
from decision_making.src.dm_main import NAVIGATION_PLAN, DEFAULT_MAP_FILE
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpaceContainer
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import RelativeLane, RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.default_config import DEFAULT_DYNAMIC_RECIPE_FILTERING, \
    DEFAULT_STATIC_RECIPE_FILTERING
from decision_making.src.planning.behavioral.evaluators.rule_based_action_spec_evaluator import \
    RuleBasedActionSpecEvaluator
from decision_making.src.planning.behavioral.evaluators.zero_value_approximator import ZeroValueApproximator
from decision_making.src.planning.behavioral.filtering.action_spec_filter_bank import FilterIfNone
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import ActionSpecFiltering
from decision_making.src.planning.behavioral.planner.single_step_behavioral_planner import SingleStepBehavioralPlanner
from decision_making.src.planning.types import FS_SX, FS_SV
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

    lane_length = MapUtils.get_lane_length(200)
    lane_ids = np.array(range(300))
    lanes_road_lon = (np.floor(lane_ids / 10) % 10) * lane_length
    ego_road_lon = ego.map_state.lane_fstate[FS_SX] + lanes_road_lon[ego.map_state.lane_id]
    obj_road_lon = np.array([obj.map_state.lane_fstate[FS_SX] + lanes_road_lon[obj.map_state.lane_id]
                             for obj in state.dynamic_objects])
    obj_lane = [obj.map_state.lane_id % 10 for obj in state.dynamic_objects]
    object_vels = np.array([obj.map_state.lane_fstate[FS_SV] for obj in state.dynamic_objects])

    action_space = ActionSpaceContainer(logger, [StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING),
                                                 DynamicActionSpace(logger, predictor,
                                                                    DEFAULT_DYNAMIC_RECIPE_FILTERING)])
    recipe_evaluator = None
    action_spec_evaluator = RuleBasedActionSpecEvaluator(logger)
    value_approximator = ZeroValueApproximator(logger)
    action_spec_filtering = ActionSpecFiltering(filters=[FilterIfNone()], logger=logger)

    planner = SingleStepBehavioralPlanner(action_space, recipe_evaluator, action_spec_evaluator,
                                          action_spec_filtering, value_approximator, predictor, logger)

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

    valid_specs = np.array([spec for i, spec in enumerate(action_specs) if action_specs_mask[i]])
    valid_terminal_states = np.array([state for state in terminal_behavioral_states if state is not None])

    ego_terminal_road_lon = np.array([state.ego_state.map_state.lane_fstate[FS_SX] +
                                      lanes_road_lon[state.ego_state.map_state.lane_id]
                                      for state in valid_terminal_states])
    ego_terminal_lane = np.array([state.ego_state.map_state.lane_id % 10 for state in valid_terminal_states])
    obj_terminal_lon = np.array([obj_road_lon + spec.t * object_vels for spec in valid_specs])
    terminal_rel_lon = obj_terminal_lon - ego_terminal_road_lon[:, np.newaxis]
    terminal_rel_lane = np.array(obj_lane*len(valid_specs)).reshape(len(valid_specs), -1) - ego_terminal_lane[:, np.newaxis]
    for i, obj in enumerate(state.dynamic_objects):
        for j, spec in enumerate(valid_specs):
            lat_cell = RelativeLane(terminal_rel_lane[j, i]) if abs(terminal_rel_lane[j, i]) <= 1 else None
            lon_cell = RelativeLongitudinalPosition.FRONT if terminal_rel_lon > 4 \
                else RelativeLongitudinalPosition.REAR if terminal_rel_lon < -4 else RelativeLongitudinalPosition.PARALLEL


    assert len(terminal_behavioral_states) == len(action_specs)
    assert len(valid_terminal_states) == len(valid_specs)
    assert np.isclose(
        np.array([state.ego_state.timestamp_in_sec for state in terminal_behavioral_states if state is not None]),
        np.array([timestamp_in_sec + spec.t for i, spec in enumerate(action_specs) if action_specs_mask[i]])).all()

    ego_lane_ordinal = MapUtils.get_lane_ordinal(ego.map_state.lane_id)
    for obj in state.dynamic_objects:
        lane_ordinal = MapUtils.get_lane_ordinal(obj.map_state.lane_id)
        rel_lane_val = lane_ordinal - ego_lane_ordinal
        if abs(rel_lane_val) <= 1:
            rel_lane = RelativeLane(rel_lane_val)
            behavioral_state.projected_ego_fstates[rel_lane]

    # verify all terminal states have non-empty road_occupancy_grid
    assert (np.array([len(state.road_occupancy_grid) for state in terminal_behavioral_states if state is not None]) > 0).all()

    for i, state in enumerate(valid_terminal_states):
        for cell, objects_with_semantics in state.road_occupancy_grid.items():
            for obj in objects_with_semantics:
                pass