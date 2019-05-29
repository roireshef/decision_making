import numpy as np
from decision_making.src.messages.scene_static_message import StaticTrafficFlowControl, RoadObjectType
from decision_making.src.planning.types import FS_SX
from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.utils.map_utils import MapUtils
from decision_making.test.messages.scene_static_fixture import scene_static_pg_split
from typing import List
from unittest.mock import patch

from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.data_objects import DynamicActionRecipe, StaticActionRecipe, ActionSpec, \
    ActionRecipe, RelativeLane, ActionType, AggressivenessLevel
from decision_making.src.planning.behavioral.filtering.action_spec_filter_bank import FilterForKinematics, \
    FilterIfNone as FilterSpecIfNone, FilterForSafetyTowardsTargetVehicle, StaticTrafficFlowControlFilter, \
    BeyondSpecStaticTrafficFlowControlFilter, BeyondSpecSpeedLimitFilter
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import ActionSpecFiltering
from decision_making.src.planning.behavioral.filtering.recipe_filter_bank import FilterIfNone as FilterRecipeIfNone
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from rte.python.logger.AV_logger import AV_Logger

from decision_making.test.planning.behavioral.behavioral_state_fixtures import \
    behavioral_grid_state_with_objects_for_acceleration_towards_vehicle, \
    behavioral_grid_state_with_objects_for_filtering_almost_tracking_mode, \
    state_with_objects_for_acceleration_towards_vehicle, \
    state_with_objects_for_filtering_exact_tracking_mode, \
    behavioral_grid_state_with_objects_for_filtering_too_aggressive, \
    state_with_objects_for_filtering_almost_tracking_mode, \
    behavioral_grid_state_with_objects_for_filtering_exact_tracking_mode, \
    state_with_objects_for_filtering_too_aggressive, follow_vehicle_recipes_towards_front_cells, follow_lane_recipes, \
    behavioral_grid_state_with_traffic_control, state_with_traffic_control, route_plan_20_30, route_plan_oval_track


def test_StaticTrafficFlowControlFilter_filtersWhenTrafficFlowControlexits(behavioral_grid_state_with_traffic_control,
                                                                           scene_static_pg_split):
    ego_location = behavioral_grid_state_with_traffic_control.ego_state.map_state.lane_fstate[FS_SX]

    gff = behavioral_grid_state_with_traffic_control.extended_lane_frames[RelativeLane.SAME_LANE]
    gff_state = np.array([[ego_location + 12.0, 0., 0., 0., 0., 0.]])
    lane_id, segment_states = gff.convert_to_segment_states(gff_state)
    segment_s = segment_states[0][0]

    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    stop_sign = StaticTrafficFlowControl(e_e_road_object_type=RoadObjectType.StopSign, e_l_station=segment_s,
                                         e_Pct_confidence=1.0)
    MapUtils.get_lane(lane_id).as_static_traffic_flow_control.append(stop_sign)

    filter = StaticTrafficFlowControlFilter()
    t, v, s, d = 10, 20, ego_location + 40.0, 0
    action_specs = [ActionSpec(t, v, s, d,
                               ActionRecipe(RelativeLane.SAME_LANE, ActionType.FOLLOW_LANE, AggressivenessLevel.CALM))]
    actual = filter.filter(action_specs=action_specs, behavioral_state=behavioral_grid_state_with_traffic_control)
    expected = [False]
    assert actual == expected


def test_BeyondSpecStaticTrafficFlowControlFilter_filtersWhenTrafficFlowControlexits(behavioral_grid_state_with_traffic_control,
                                                                                     scene_static_pg_split):

    ego_location = behavioral_grid_state_with_traffic_control.ego_state.map_state.lane_fstate[FS_SX]
    gff = behavioral_grid_state_with_traffic_control.extended_lane_frames[RelativeLane.SAME_LANE]

    gff_stop_sign_location = ego_location + 42.0

    gff_state = np.array([[gff_stop_sign_location, 0., 0., 0., 0., 0.]])
    lane_id, segment_states = gff.convert_to_segment_states(gff_state)
    segment_s = segment_states[0][0]

    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    stop_sign = StaticTrafficFlowControl(e_e_road_object_type=RoadObjectType.StopSign, e_l_station=segment_s,
                                         e_Pct_confidence=1.0)
    MapUtils.get_lane(lane_id).as_static_traffic_flow_control.append(stop_sign)

    filter = BeyondSpecStaticTrafficFlowControlFilter()
    t, v, s, d = 10, 20, gff_stop_sign_location - 2.0, 0
    action_specs = [ActionSpec(t, v, s, d,
                               ActionRecipe(RelativeLane.SAME_LANE, ActionType.FOLLOW_LANE, AggressivenessLevel.CALM))]
    actual = filter.filter(action_specs=action_specs, behavioral_state=behavioral_grid_state_with_traffic_control)
    expected = [False]
    assert actual == expected


def test_BeyondSpecSpeedLimitFilter_SlowLaneAhead(behavioral_grid_state_with_traffic_control, scene_static_pg_split):
    # Get s position on frenet frame
    ego_location = behavioral_grid_state_with_traffic_control.ego_state.map_state.lane_fstate[FS_SX]
    gff = behavioral_grid_state_with_traffic_control.extended_lane_frames[RelativeLane.SAME_LANE]

    gff_states_up_to_speed_limit = np.array(
        [[np.float(i), 0., 0., 0., 0., 0.] for i in range(int(ego_location), int(ego_location) + 3)])

    # put some slow speed limits into scene_static
    for i in range(1, 23):
        scene_static_pg_split.s_Data.s_SceneStaticBase.as_scene_lane_segments[-i].e_v_nominal_speed = 10

    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)

    filter = BeyondSpecSpeedLimitFilter()
    t, v, s, d = 10, 34, ego_location + 80, 0
    action_specs = [
        ActionSpec(t, v, s, d, ActionRecipe(RelativeLane.SAME_LANE, ActionType.FOLLOW_LANE, AggressivenessLevel.CALM))]
    actual = filter.filter(action_specs=action_specs, behavioral_state=behavioral_grid_state_with_traffic_control)
    expected = [False]
    assert actual == expected


def test_BeyondSpecSpeedLimitFilter_NoSpeedLimitChange(behavioral_grid_state_with_traffic_control, scene_static_pg_split):
    # Get s position on frenet frame
    ego_location = behavioral_grid_state_with_traffic_control.ego_state.map_state.lane_fstate[FS_SX]
    gff = behavioral_grid_state_with_traffic_control.extended_lane_frames[RelativeLane.SAME_LANE]

    gff_states_up_to_speed_limit = np.array(
        [[np.float(i), 0., 0., 0., 0., 0.] for i in range(int(ego_location), int(ego_location) + 3)])

    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)

    filter = BeyondSpecSpeedLimitFilter()
    t, v, s, d = 10, 34, ego_location + 80, 0
    action_specs = [
        ActionSpec(t, v, s, d, ActionRecipe(RelativeLane.SAME_LANE, ActionType.FOLLOW_LANE, AggressivenessLevel.CALM))]
    actual = filter.filter(action_specs=action_specs, behavioral_state=behavioral_grid_state_with_traffic_control)
    expected = [True]
    assert actual == expected


@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT', 5)
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.SAFETY_HEADWAY', 0.7)
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.LON_ACC_LIMITS', np.array([-5.5, 3.0]))
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.LAT_ACC_LIMITS', np.array([-4.0, 4.0]))
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED', 40 / 3.6)
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT', 5.0)
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.SPECIFICATION_HEADWAY', 1.5)
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.BP_ACTION_T_LIMITS', np.array([0, 15]))
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.BP_JERK_S_JERK_D_TIME_WEIGHTS', np.array([
    [12, 0.15, 0.1],
    [2, 0.15, 0.1],
    [0.01, 0.15, 0.1]
]))
def test_filter_accelerationTowardsVehicle_filterResultsMatchExpected(
        behavioral_grid_state_with_objects_for_acceleration_towards_vehicle,
        follow_vehicle_recipes_towards_front_cells: List[DynamicActionRecipe]):
    """ see velocities and accelerations at https://www.desmos.com/calculator/betept6wyx """
    logger = AV_Logger.get_logger()
    predictor = RoadFollowingPredictor(logger)

    filtering = RecipeFiltering(filters=[], logger=logger)

    # only look at the same lane, front cell actions
    actions_with_vehicle = follow_vehicle_recipes_towards_front_cells[3:6]

    expected_filter_results = np.array([False, True, False], dtype=bool)
    dynamic_action_space = DynamicActionSpace(logger, predictor, filtering=filtering)

    action_specs_with_vehicle = dynamic_action_space.specify_goals(actions_with_vehicle,
                                                                   behavioral_grid_state_with_objects_for_acceleration_towards_vehicle)

    action_spec_filter = ActionSpecFiltering(filters=[FilterSpecIfNone(), FilterForKinematics()], logger=logger)

    filter_results = action_spec_filter.filter_action_specs(action_specs_with_vehicle,
                                                            behavioral_grid_state_with_objects_for_acceleration_towards_vehicle)

    np.testing.assert_array_equal(filter_results, expected_filter_results)



@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT', 5)
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.SAFETY_HEADWAY', 0.7)
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.LON_ACC_LIMITS', np.array([-5.5, 3.0]))
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.LAT_ACC_LIMITS', np.array([-4.0, 4.0]))
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED', 100 / 3.6)
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT', 5.0)
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.SPECIFICATION_HEADWAY', 1.5)
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.BP_ACTION_T_LIMITS', np.array([0, 15]))
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.BP_JERK_S_JERK_D_TIME_WEIGHTS', np.array([
    [12, 0.15, 0.1],
    [2, 0.15, 0.1],
    [0.01, 0.15, 0.1]
]))
def test_filter_closeToTrackingMode_allActionsAreValid(
        behavioral_grid_state_with_objects_for_filtering_almost_tracking_mode,
        follow_vehicle_recipes_towards_front_cells: List[DynamicActionRecipe]):
    """ see velocities and accelerations at https://www.desmos.com/calculator/betept6wyx """

    logger = AV_Logger.get_logger()
    predictor = RoadFollowingPredictor(logger)

    filtering = RecipeFiltering(filters=[], logger=logger)

    # only look at the same lane, front cell actions
    actions_with_vehicle = follow_vehicle_recipes_towards_front_cells[3:6]

    expected_filter_results = np.array([True, True, True], dtype=bool)
    dynamic_action_space = DynamicActionSpace(logger, predictor, filtering=filtering)

    action_specs_with_vehicle = dynamic_action_space.specify_goals(actions_with_vehicle,
                                                                   behavioral_grid_state_with_objects_for_filtering_almost_tracking_mode)

    action_spec_filter = ActionSpecFiltering(filters=[FilterSpecIfNone(), FilterForKinematics()], logger=logger)

    filter_results = action_spec_filter.filter_action_specs(action_specs_with_vehicle,
                                                            behavioral_grid_state_with_objects_for_filtering_almost_tracking_mode)

    np.testing.assert_array_equal(filter_results, expected_filter_results)


@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT', 5)
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.SAFETY_HEADWAY', 0.7)
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.LON_ACC_LIMITS', np.array([-5.5, 3.0]))
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.LAT_ACC_LIMITS', np.array([-4.0, 4.0]))
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT', 5.0)
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.SPECIFICATION_HEADWAY', 1.5)
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.BP_ACTION_T_LIMITS', np.array([0, 15]))
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.BP_JERK_S_JERK_D_TIME_WEIGHTS', np.array([
    [12, 0.15, 0.1],
    [2, 0.15, 0.1],
    [0.01, 0.15, 0.1]
]))
def test_filter_trackingMode_allActionsAreValid(
        behavioral_grid_state_with_objects_for_filtering_exact_tracking_mode,
        follow_vehicle_recipes_towards_front_cells: List[DynamicActionRecipe]):

    logger = AV_Logger.get_logger()
    predictor = RoadFollowingPredictor(logger)

    filtering = RecipeFiltering(filters=[], logger=logger)

    # only look at the same lane, front cell actions
    actions_with_vehicle = follow_vehicle_recipes_towards_front_cells[3:6]

    expected_filter_results = np.array([True, True, True], dtype=bool)
    dynamic_action_space = DynamicActionSpace(logger, predictor, filtering=filtering)

    action_specs_with_vehicle = dynamic_action_space.specify_goals(actions_with_vehicle,
                                                                   behavioral_grid_state_with_objects_for_filtering_exact_tracking_mode)

    action_spec_filter = ActionSpecFiltering(filters=[FilterSpecIfNone(), FilterForKinematics()], logger=logger)

    filter_results = action_spec_filter.filter_action_specs(action_specs_with_vehicle,
                                                            behavioral_grid_state_with_objects_for_filtering_exact_tracking_mode)

    np.testing.assert_array_equal(filter_results, expected_filter_results)


@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT', 5)
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.SAFETY_HEADWAY', 0.7)
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.LON_ACC_LIMITS', np.array([-5.5, 3.0]))
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.LAT_ACC_LIMITS', np.array([-4.0, 4.0]))
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT', 5.0)
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.SPECIFICATION_HEADWAY', 1.5)
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.BP_ACTION_T_LIMITS', np.array([0, 15]))
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.BP_JERK_S_JERK_D_TIME_WEIGHTS', np.array([
    [12, 0.15, 0.1],
    [2, 0.15, 0.1],
    [0.01, 0.15, 0.1]
]))
def test_filter_staticActionsWithLeadingVehicle_filterResultsMatchExpected(
        behavioral_grid_state_with_objects_for_filtering_almost_tracking_mode,
        follow_lane_recipes: List[StaticActionRecipe]):
    """
    # actions [0, 1, 3, 9, 12, 13, 15, 16] are None after specify
    # actions [6-17] are static, aiming to higher velocity - which hits the front vehicle
    # action 8 can be seen here: https://www.desmos.com/calculator/dtntkm1hsr
    """

    logger = AV_Logger.get_logger()
    predictor = RoadFollowingPredictor(logger)

    filtering = RecipeFiltering(filters=[], logger=logger)

    expected_filter_results = np.array([False, False, True, False, True, True, False, False, True,
                                        False, False, False, False, False, False, False, False, False], dtype=bool)
    static_action_space = StaticActionSpace(logger, filtering=filtering)

    action_specs = static_action_space.specify_goals(follow_lane_recipes,
                                                     behavioral_grid_state_with_objects_for_filtering_almost_tracking_mode)

    action_spec_filter = ActionSpecFiltering(filters=[FilterSpecIfNone(), FilterForSafetyTowardsTargetVehicle()], logger=logger)

    filter_results = action_spec_filter.filter_action_specs(action_specs,
                                                            behavioral_grid_state_with_objects_for_filtering_almost_tracking_mode)

    # TODO: action 8 is True because FilterForSafetyTowardsTargetVehicle doesn't check the padding after a short horizon
    np.testing.assert_array_equal(filter_results, expected_filter_results)


@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT', 5)
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.SAFETY_HEADWAY', 0.7)
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.LON_ACC_LIMITS', np.array([-5.5, 3.0]))
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.LAT_ACC_LIMITS', np.array([-4.0, 4.0]))
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT', 5.0)
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.SPECIFICATION_HEADWAY', 1.5)
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.BP_ACTION_T_LIMITS', np.array([0, 15]))
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.BP_JERK_S_JERK_D_TIME_WEIGHTS', np.array([
    [12, 0.15, 0.1],
    [2, 0.15, 0.1],
    [0.01, 0.15, 0.1]
]))
def test_filter_aggressiveFollowScenario_allActionsAreInvalid(
        behavioral_grid_state_with_objects_for_filtering_too_aggressive,
        follow_vehicle_recipes_towards_front_cells: List[DynamicActionRecipe]):
    """
    State leads to a0=0,v0=10,sT=53.5,vT=30
    Results are false because this following scenario is too aggressive (close a gap of 20[m/s] to a target vehicle ~50[m] from us)
    calm and standard actions take too much time and aggressive violates velocity and acceleration
    All ground truths checked with desmos - https://www.desmos.com/calculator/exizg3iuhs
    """
    logger = AV_Logger.get_logger()
    predictor = RoadFollowingPredictor(logger)

    filtering = RecipeFiltering(filters=[FilterRecipeIfNone()], logger=logger)

    actions_with_vehicle = follow_vehicle_recipes_towards_front_cells[3:6]


    expected_filter_results = np.array([False, False, False], dtype=bool)
    dynamic_action_space = DynamicActionSpace(logger, predictor, filtering=filtering)

    action_specs_with_vehicle = dynamic_action_space.specify_goals(actions_with_vehicle,
                                                                   behavioral_grid_state_with_objects_for_filtering_too_aggressive)

    action_spec_filter = ActionSpecFiltering(filters=[FilterSpecIfNone(), FilterForKinematics()], logger=logger)

    filter_results = action_spec_filter.filter_action_specs(action_specs_with_vehicle,
                                                            behavioral_grid_state_with_objects_for_filtering_too_aggressive)

    np.testing.assert_array_equal(filter_results, expected_filter_results)