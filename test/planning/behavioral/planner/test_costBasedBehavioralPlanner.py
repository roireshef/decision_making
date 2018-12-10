import numpy as np
from decision_making.src.global_constants import LON_MARGIN_FROM_EGO, PLANNING_LOOKAHEAD_DIST
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
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
from decision_making.test.messages.static_scene_fixture import create_scene_static_from_map_api
from mapping.src.model.map_api import MapAPI
from mapping.test.model.map_model_utils import TestMapModelUtils
from rte.python.logger.AV_logger import AV_Logger


def test_generateTerminalStates_multiRoad_accurateGridCells():
    """
    Test the function _generate_terminal_states for a custom map (straight road) with 5 lanes, 10 road segments
    and 9 dynamic objects on all lanes.
    For each terminal state validate the number of the grid cells and their content by creating an exact ground truth.
    """
    logger = AV_Logger.get_logger()
    x_coord = np.arange(0., 1000.1, 0.5)
    road_points = np.c_[x_coord, np.full(x_coord.shape[0], 0)]
    lane_width = 4
    num_lanes = 5
    single_seg_map_model = TestMapModelUtils.create_road_map_from_coordinates(points_of_roads=[road_points],
                                                                              road_id=[1],
                                                                              road_name=['def'],
                                                                              lanes_num=[num_lanes],
                                                                              lane_width=[lane_width],
                                                                              frame_origin=[0, 0])
    test_map_model = TestMapModelUtils.split_road(single_seg_map_model, 10)
    map_api = MapAPI(map_model=test_map_model, logger=logger)
    scene_static = create_scene_static_from_map_api(map_api)
    SceneStaticModel.get_instance().set_scene_static(scene_static)

    predictor = RoadFollowingPredictor(logger)
    nav_plan = NavigationPlanMsg(np.array(test_map_model.get_road_ids()))
    size = ObjectSize(4, 2, 1)
    timestamp_in_sec = 1000
    timestamp = int(timestamp_in_sec * 1e9)

    # Create the initial state

    # lane ordinal = 1, road_segment = 1 (the first one)
    ego_cstate = MapUtils.get_lane_frenet_frame(11).fstate_to_cstate(np.array([60, 14, 0, 0, 0, 0]))
    ego = EgoState.create_from_cartesian_state(obj_id=0, timestamp=timestamp, cartesian_state=ego_cstate, size=size,
                                               confidence=0)
    # (same_lane, front) cell
    obj1_cstate = MapUtils.get_lane_frenet_frame(11).fstate_to_cstate(np.array([80, 14, 0, 0, 0, 0]))
    obj1 = EgoState.create_from_cartesian_state(obj_id=1, timestamp=timestamp, cartesian_state=obj1_cstate,
                                                size=size, confidence=0)
    # (right_lane, front) cell
    obj2_cstate = MapUtils.get_lane_frenet_frame(10).fstate_to_cstate(np.array([70, 10, 0, 0, 0, 0]))
    obj2 = EgoState.create_from_cartesian_state(obj_id=2, timestamp=timestamp, cartesian_state=obj2_cstate,
                                                size=size, confidence=0)
    # (left_lane, front) cell
    obj3_cstate = MapUtils.get_lane_frenet_frame(12).fstate_to_cstate(np.array([90, 16, 0, 0, 0, 0]))
    obj3 = EgoState.create_from_cartesian_state(obj_id=3, timestamp=timestamp, cartesian_state=obj3_cstate,
                                                size=size, confidence=0)
    # (same_lane, front) cell
    obj4_cstate = MapUtils.get_lane_frenet_frame(21).fstate_to_cstate(np.array([5, 17, 0, 0, 0, 0]))
    obj4 = EgoState.create_from_cartesian_state(obj_id=4, timestamp=timestamp, cartesian_state=obj4_cstate,
                                                size=size, confidence=0)
    # (left_lane, front) cell
    obj5_cstate = MapUtils.get_lane_frenet_frame(22).fstate_to_cstate(np.array([10, 15, 0, 0, 0, 0]))
    obj5 = EgoState.create_from_cartesian_state(obj_id=5, timestamp=timestamp, cartesian_state=obj5_cstate,
                                                size=size, confidence=0)
    # (right_lane, rear) cell
    obj6_cstate = MapUtils.get_lane_frenet_frame(10).fstate_to_cstate(np.array([30, 15, 0, 0, 0, 0]))
    obj6 = EgoState.create_from_cartesian_state(obj_id=6, timestamp=timestamp, cartesian_state=obj6_cstate,
                                                size=size, confidence=0)
    # (left of left_lane, rear) cell
    obj7_cstate = MapUtils.get_lane_frenet_frame(13).fstate_to_cstate(np.array([50, 12, 0, 0, 0, 0]))
    obj7 = EgoState.create_from_cartesian_state(obj_id=7, timestamp=timestamp, cartesian_state=obj7_cstate,
                                                size=size, confidence=0)
    # (left of left_lane, front) cell
    obj8_cstate = MapUtils.get_lane_frenet_frame(23).fstate_to_cstate(np.array([20, 16, 0, 0, 0, 0]))
    obj8 = EgoState.create_from_cartesian_state(obj_id=8, timestamp=timestamp, cartesian_state=obj8_cstate,
                                                size=size, confidence=0)
    # (left of left of left, parallel) cell
    obj9_cstate = MapUtils.get_lane_frenet_frame(14).fstate_to_cstate(np.array([60, 16, 0, 0, 0, 0]))
    obj9 = EgoState.create_from_cartesian_state(obj_id=9, timestamp=timestamp, cartesian_state=obj9_cstate,
                                                size=size, confidence=0)

    state = State(None, [obj1, obj2, obj3, obj4, obj5, obj6, obj7, obj8, obj9], ego)

    lane_length = MapUtils.get_lane_length(10)
    lane_ids = np.array(range(100))
    lanes_lon_offset = (np.floor((lane_ids - 10) / 10) % 10) * lane_length
    ego_lane_ordinal = state.ego_state.map_state.lane_id % 10
    obj_road_lon = np.array([obj.map_state.lane_fstate[FS_SX] + lanes_lon_offset[obj.map_state.lane_id]
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

    # generate terminal states
    terminal_behavioral_states = planner._generate_terminal_states(state, behavioral_state, action_specs,
                                                                   action_specs_mask, nav_plan)

    valid_specs = np.array([spec for i, spec in enumerate(action_specs) if action_specs_mask[i]])
    valid_terminal_states = np.array([state for state in terminal_behavioral_states if state is not None])

    # validate the number of all terminal states
    assert len(terminal_behavioral_states) == len(action_specs)
    # validate the number of valid terminal states and their timestamps
    assert len(valid_terminal_states) == len(valid_specs)
    assert np.isclose(
        np.array([state.ego_state.timestamp_in_sec for state in terminal_behavioral_states if state is not None]),
        np.array([timestamp_in_sec + spec.t for i, spec in enumerate(action_specs) if action_specs_mask[i]])).all()

    # create the GROUND TRUTH for all terminal behavioral states
    ego_terminal_lon = np.array([spec.s for spec in valid_specs])
    ego_terminal_lane = np.array([ego_lane_ordinal + spec.relative_lane.value for spec in valid_specs])
    obj_terminal_lon = np.array([obj_road_lon + spec.t * object_vels for spec in valid_specs])
    terminal_rel_lon = obj_terminal_lon - ego_terminal_lon[:, np.newaxis]
    terminal_rel_lane = np.array(obj_lane*len(valid_specs)).reshape(len(valid_specs), -1) - ego_terminal_lane[:, np.newaxis]
    full_grid = []
    parallel_lon_thresh = state.dynamic_objects[0].size.length / 2 + state.ego_state.size.length / 2 + LON_MARGIN_FROM_EGO
    max_lon = behavioral_state.extended_lane_frames[RelativeLane.SAME_LANE].s_max
    for j, spec in enumerate(valid_specs):
        grid = {}
        for i, obj in enumerate(state.dynamic_objects):
            rel_lon = terminal_rel_lon[j, i]
            if abs(rel_lon) > PLANNING_LOOKAHEAD_DIST or obj_terminal_lon[j, i] > max_lon or abs(terminal_rel_lane[j, i]) > 1:
                continue
            lat_cell = RelativeLane(terminal_rel_lane[j, i])
            lon_cell = RelativeLongitudinalPosition.FRONT if rel_lon > parallel_lon_thresh \
                else RelativeLongitudinalPosition.REAR if rel_lon < -parallel_lon_thresh else RelativeLongitudinalPosition.PARALLEL
            if (lat_cell, lon_cell) not in grid or abs(grid[(lat_cell, lon_cell)][1]) > abs(rel_lon):
                grid[(lat_cell, lon_cell)] = (obj.obj_id, rel_lon)
        full_grid.append(grid)

    # validate for all terminal states:
    for j, state in enumerate(valid_terminal_states):
        # validate terminal s
        assert np.isclose(ego_terminal_lon[j], state.ego_state.map_state.lane_fstate[FS_SX] + lanes_lon_offset[state.ego_state.map_state.lane_id])
        # validate terminal lane ordinal
        assert ego_terminal_lane[j] == state.ego_state.map_state.lane_id % 10
        # validate number of grid cells
        assert len(state.road_occupancy_grid) == len(full_grid[j])
        # validate the content of all grid cells (objects ids & longitudinal distance)
        for cell, objects in state.road_occupancy_grid.items():
            assert objects[0].dynamic_object.obj_id == full_grid[j][(cell[0], cell[1])][0]
            assert np.isclose(objects[0].longitudinal_distance, full_grid[j][(cell[0], cell[1])][1])
