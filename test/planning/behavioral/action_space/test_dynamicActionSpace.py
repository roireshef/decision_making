from decision_making.src.scene.scene_static_model import SceneStaticModel
from logging import Logger
from typing import List
import pdb
import copy
import numpy as np

from decision_making.src.global_constants import SPECIFICATION_MARGIN_TIME_DELAY, \
    LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, LON_ACC_LIMITS, BP_ACTION_T_LIMITS
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import DynamicActionRecipe, ActionSpec, RelativeLane
from decision_making.src.planning.behavioral.default_config import DEFAULT_DYNAMIC_RECIPE_FILTERING
from decision_making.src.planning.trajectory.werling_planner import WerlingPlanner
from decision_making.src.planning.types import FS_SX, FS_SV, FS_SA
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D, Poly1D
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import ObjectSize, EgoState, DynamicObject, State
from decision_making.src.utils.map_utils import MapUtils
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from rte.python.logger.AV_logger import AV_Logger

from decision_making.test.planning.behavioral.behavioral_state_fixtures import behavioral_grid_state, \
    follow_vehicle_recipes_towards_front_cells, state_with_sorrounding_objects, \
    follow_vehicle_recipes_towards_front_same_lane
from decision_making.test.messages.static_scene_fixture import scene_static

NAVIGATION_PLAN = NavigationPlanMsg(np.array(range(20, 30)))


# Specifies follow actions for front vehicles in 3 lanes. longitudinal and lateral coordinates
# of terminal states in action specification should be as expected.
# Multi-segment map is used, such that the targets have different road segment than ego.
def test_specifyGoals_stateWithSorroundingObjects_specifiesFollowTowardsFrontCellsWell(
        behavioral_grid_state: BehavioralGridState,
        follow_vehicle_recipes_towards_front_cells: List[DynamicActionRecipe]):
    logger = AV_Logger.get_logger()
    predictor = RoadFollowingPredictor(logger)  # TODO: adapt to new changes

    dynamic_action_space = DynamicActionSpace(logger, predictor, filtering=DEFAULT_DYNAMIC_RECIPE_FILTERING)
    actions = dynamic_action_space.specify_goals(follow_vehicle_recipes_towards_front_cells, behavioral_grid_state)

    targets = [behavioral_grid_state.road_occupancy_grid[(recipe.relative_lane, recipe.relative_lon)][0]
               for recipe in follow_vehicle_recipes_towards_front_cells]

    # terminal action-spec latitude equals the current latitude of target vehicle
    expected_latitudes = [0]*9
    latitudes = [action.d for i, action in enumerate(actions)]
    np.testing.assert_array_almost_equal(latitudes, expected_latitudes)

    # since the map is multi-segment, calculate objects longitudes relative to the corresponding GFF
    ego_ordinal = MapUtils.get_lane_ordinal(behavioral_grid_state.ego_state.map_state.lane_id)
    objects_longitudes = []
    for i, target in enumerate(targets):
        target_map_state = target.dynamic_object.map_state
        target_ordinal = MapUtils.get_lane_ordinal(target_map_state.lane_id)
        rel_lane = RelativeLane(target_ordinal - ego_ordinal)
        target_gff_fstate = behavioral_grid_state.extended_lane_frames[rel_lane].convert_from_segment_state(
            target_map_state.lane_fstate, target_map_state.lane_id)
        objects_longitudes.append(target_gff_fstate[FS_SX])

    # terminal action-spec longitude equals the terminal longitude of target vehicle
    # (according to prediction at the terminal time)
    expected_longitudes = [objects_longitudes[i] +
                           target.dynamic_object.map_state.lane_fstate[FS_SV] * actions[i].t -
                           actions[i].v * SPECIFICATION_MARGIN_TIME_DELAY -
                           LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT -
                           behavioral_grid_state.ego_state.size.length / 2 - targets[i].dynamic_object.size.length / 2
                           for i, target in enumerate(targets)]

    longitudes = [action.s for action in actions]  # also relative to the corresponding GFF
    np.testing.assert_array_almost_equal(longitudes, expected_longitudes)


def test_specifyGoals_followSlowFrontObject_specsComplyAccAndTimeLimits(scene_static,
        follow_vehicle_recipes_towards_front_same_lane: List[DynamicActionRecipe]):
    """
    Test specify_goals for dynamic actions towards much slower front object.
    Verify that choosing of maximal roots of 'time_cost_function_derivative_coefs' polynomial create specs
    that comply with acceleration and time limits, while choosing minimal roots does not comply the limits.
    :param follow_vehicle_recipes_towards_front_same_lane:
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static)

    logger = AV_Logger.get_logger()
    predictor = RoadFollowingPredictor(logger)
    dynamic_action_space = DynamicActionSpace(logger, predictor, filtering=DEFAULT_DYNAMIC_RECIPE_FILTERING)
    road_id = MapUtils.get_road_segment_ids()[0]
    lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_id)[0]

    ego_init_fstate = np.array([13.3209042, 8.3501435, 0.08652155, 1.8, 0, 0])
    obj_map_state = MapState(np.array([29.7766345, 2.77391842, 0, 1.8, 0, 0]), lane_id)

    ego_map_state = MapState(ego_init_fstate, lane_id)
    ego = EgoState.create_from_map_state(0, 0, ego_map_state, ObjectSize(5, 2, 0), 0)
    obj = DynamicObject.create_from_map_state(1, 0, obj_map_state, ObjectSize(2.66, 1.33, 0), 0)
    state = State(None, [obj], ego)
    behavioral_state = BehavioralGridState.create_from_state(state, NAVIGATION_PLAN, logger)

    actions = dynamic_action_space.specify_goals(follow_vehicle_recipes_towards_front_same_lane, behavioral_state)
    specs = [spec for spec in actions if spec is not None]

    acc_in_limits = np.zeros(len(specs))
    time_in_limits = np.zeros(len(specs))
    for spec_idx, spec in enumerate(specs):
        print('spec = %s' % spec)
        time_samples = np.arange(0, spec.t, 0.1)
        constraints_s = np.array([[ego_init_fstate[FS_SX], ego_init_fstate[FS_SV], ego_init_fstate[FS_SA], spec.s, spec.v, 0]])
        poly_s = WerlingPlanner._solve_1d_poly(constraints_s, spec.t, QuinticPoly1D)
        ftraj = QuinticPoly1D.polyval_with_derivatives(np.array(poly_s), time_samples)[0]
        acc_in_limits[spec_idx] = np.logical_and(LON_ACC_LIMITS[0] < ftraj[:, FS_SA], ftraj[:, FS_SA] < LON_ACC_LIMITS[1]).all()
        time_in_limits[spec_idx] = spec.t < BP_ACTION_T_LIMITS[1]

    assert np.any(np.logical_and(acc_in_limits, time_in_limits))
