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
    BeyondSpecStaticTrafficFlowControlFilter, FilterForLaneSpeedLimits, BeyondSpecSpeedLimitFilter, FilterForSLimit
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
    behavioral_grid_state_with_segments_limits, \
    state_for_testing_lanes_speed_limits_violations, \
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
    for i in range(scene_static_pg_split.s_Data.s_SceneStaticBase.e_Cnt_num_lane_segments):
        scene_static_pg_split.s_Data.s_SceneStaticBase.as_scene_lane_segments[-i].e_v_nominal_speed = 10

    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)

    filter = BeyondSpecSpeedLimitFilter()
    t, v, s, d = 10, 25, ego_location + 5, 0
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
    t, v, s, d = 10, 34/3.6, ego_location + 80, 0
    action_specs = [
        ActionSpec(t, v, s, d, ActionRecipe(RelativeLane.SAME_LANE, ActionType.FOLLOW_LANE, AggressivenessLevel.CALM))]
    actual = filter.filter(action_specs=action_specs, behavioral_state=behavioral_grid_state_with_traffic_control)
    expected = [True]
    assert actual == expected


@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT', 5)
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.SAFETY_HEADWAY', 0.7)
@patch.multiple('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank',LON_ACC_LIMITS=np.array([-5.5, 3.0]))
@patch.multiple('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank',LAT_ACC_LIMITS=np.array([-4.0, 4.0]))
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.FilterForLaneSpeedLimits._pointwise_nominal_speed', lambda *args : 40 / 3.6)
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT', 5.0)
@patch('decision_making.src.planning.behavioral.action_space.target_action_space.SPECIFICATION_HEADWAY', 1.5)
@patch.multiple('decision_making.src.planning.behavioral.action_space.target_action_space',BP_ACTION_T_LIMITS=np.array([0, 15]))
@patch.multiple('decision_making.src.planning.behavioral.action_space.target_action_space',BP_JERK_S_JERK_D_TIME_WEIGHTS=np.array([
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

    action_spec_filter = ActionSpecFiltering(filters=[FilterSpecIfNone(), FilterForKinematics(), FilterForLaneSpeedLimits()], logger=logger)

    filter_results = action_spec_filter.filter_action_specs(action_specs_with_vehicle,
                                                            behavioral_grid_state_with_objects_for_acceleration_towards_vehicle)

    np.testing.assert_array_equal(filter_results, expected_filter_results)


@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT', 5)
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.SAFETY_HEADWAY', 0.7)
@patch.multiple('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank',LON_ACC_LIMITS=np.array([-5.5, 3.0]))
@patch.multiple('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank',LAT_ACC_LIMITS=np.array([-4.0, 4.0]))
# @patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED', 100 / 3.6)
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT', 5.0)
@patch('decision_making.src.planning.behavioral.action_space.target_action_space.SPECIFICATION_HEADWAY', 1.5)
@patch.multiple('decision_making.src.planning.behavioral.action_space.target_action_space',BP_ACTION_T_LIMITS=np.array([0, 15]))
@patch.multiple('decision_making.src.planning.behavioral.action_space.target_action_space',BP_JERK_S_JERK_D_TIME_WEIGHTS=np.array([
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
@patch.multiple('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank',LON_ACC_LIMITS=np.array([-5.5, 3.0]))
@patch.multiple('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank',LAT_ACC_LIMITS=np.array([-4.0, 4.0]))
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT', 5.0)
@patch('decision_making.src.planning.behavioral.action_space.target_action_space.SPECIFICATION_HEADWAY', 1.5)
@patch.multiple('decision_making.src.planning.behavioral.action_space.target_action_space',BP_ACTION_T_LIMITS=np.array([0, 15]))
@patch.multiple('decision_making.src.planning.behavioral.action_space.target_action_space',BP_JERK_S_JERK_D_TIME_WEIGHTS=np.array([
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
@patch.multiple('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank',LON_ACC_LIMITS=np.array([-5.5, 3.0]))
@patch.multiple('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank',LAT_ACC_LIMITS=np.array([-4.0, 4.0]))
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT', 5.0)
@patch('decision_making.src.planning.behavioral.action_space.target_action_space.SPECIFICATION_HEADWAY', 1.5)
@patch.multiple('decision_making.src.planning.behavioral.action_space.target_action_space',BP_ACTION_T_LIMITS=np.array([0, 15]))
@patch.multiple('decision_making.src.planning.behavioral.action_space.target_action_space',BP_JERK_S_JERK_D_TIME_WEIGHTS=np.array([
    [12, 0.15, 0.1],
    [2, 0.15, 0.1],
    [0.01, 0.15, 0.1]
]))
def test_filter_staticActionsWithLeadingVehicle_filterResultsMatchExpected(
        behavioral_grid_state_with_objects_for_filtering_almost_tracking_mode,
        follow_lane_recipes: List[StaticActionRecipe]):
    """
    # actions [9, 12, 15] are None after specify
    # actions [6-17] are static, aiming to higher velocity - which hits the front vehicle
    # actions 7, 8 are safe and can be seen here: https://www.desmos.com/calculator/dtntkm1hsr
    """

    logger = AV_Logger.get_logger()

    filtering = RecipeFiltering(filters=[], logger=logger)

    expected_filter_results = np.array([True, True, True, True, True, True, False, True, True,
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
@patch.multiple('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank',LON_ACC_LIMITS=np.array([-5.5, 3.0]))
@patch.multiple('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank',LAT_ACC_LIMITS=np.array([-4.0, 4.0]))
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT', 5.0)
@patch('decision_making.src.planning.behavioral.action_space.target_action_space.SPECIFICATION_HEADWAY', 1.5)
@patch.multiple('decision_making.src.planning.behavioral.action_space.target_action_space',BP_ACTION_T_LIMITS=np.array([0, 15]))
@patch.multiple('decision_making.src.planning.behavioral.action_space.target_action_space',BP_JERK_S_JERK_D_TIME_WEIGHTS=np.array([
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


@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT', 5)
@patch('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank.SAFETY_HEADWAY', 0.7)
@patch.multiple('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank',LON_ACC_LIMITS=np.array([-5.5, 3.0]))
@patch.multiple('decision_making.src.planning.behavioral.filtering.action_spec_filter_bank',LAT_ACC_LIMITS=np.array([-4.0, 4.0]))
@patch('decision_making.src.planning.behavioral.action_space.dynamic_action_space.LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT', 5.0)
@patch('decision_making.src.planning.behavioral.action_space.target_action_space.SPECIFICATION_HEADWAY', 1.5)
@patch.multiple('decision_making.src.planning.behavioral.action_space.target_action_space',BP_ACTION_T_LIMITS=np.array([0, 15]))
@patch.multiple('decision_making.src.planning.behavioral.action_space.target_action_space',BP_JERK_S_JERK_D_TIME_WEIGHTS=np.array([
    [12, 0.15, 0.1],
    [2, 0.15, 0.1],
    [0.01, 0.15, 0.1]
]))
def test_filter_laneSpeedLimits_filtersSpecsViolatingLaneSpeedLimits_filterResultsMatchExpected(
        behavioral_grid_state_with_segments_limits,
        follow_lane_recipes: List[StaticActionRecipe]):

    logger = AV_Logger.get_logger()
    # The scene_static that is being used, is in accordance to whatever happens in behavioral_grid_state_
    # with_segments_limits fixture
    scene_static_with_limits = SceneStaticModel.get_instance().get_scene_static()
    # The following are 4 consecutive lane segments with varying speed limits (ego starts at the end of [0])
    # These are the s-values that correspond to lane transitions on the GFF:
    # [0.0, 100.84134201631973, 220.48438762415998, 343.9575891327402, 466.0989153990629]
    scene_static_with_limits.s_Data.s_SceneStaticBase.as_scene_lane_segments[0].e_v_nominal_speed = 25
    scene_static_with_limits.s_Data.s_SceneStaticBase.as_scene_lane_segments[3].e_v_nominal_speed = 25
    scene_static_with_limits.s_Data.s_SceneStaticBase.as_scene_lane_segments[6].e_v_nominal_speed = 15
    scene_static_with_limits.s_Data.s_SceneStaticBase.as_scene_lane_segments[9].e_v_nominal_speed = 25
    SceneStaticModel.get_instance().set_scene_static(scene_static_with_limits)

    filtering = RecipeFiltering(filters=[], logger=logger)
    # note: first lane segment speed limit is almost irrelevant because we start at the end of this segment
    expected_filter_results = np.array([True, True, True,   # v_T=0 (Calm, Standard, Aggressive)  - All Pass (not arriving at [9])
                                        True, True, True,   # v_T=6 (Calm, Standard, Aggressive)  - All Pass (not arriving at [9])
                                        True, True, True,   # v_T=12 (Calm, Standard, Aggressive) - All Pass (not arriving at [9])
                                        False,              # This one (calm) supposed to end at the third ([6]) lane segment with v_T=18>15
                                        True, True,         # (std,agg) v_T=18 - Pass (not arriving at [6])
                                        False,              # (calm) action_spec is None
                                        False,              # (std.) This one supposed to end at the third ([6]) lane segment with v_T=24>15
                                        True,               # (agg.) v_T=24  - Pass (not arriving at [6])
                                        False,              # (calm) action_spec is None
                                        False,              # (std.) This one supposed to end at the fourth ([9]) lane segment with v_T=30>15
                                        False               # (agg.) This one supposed to end at the third ([6]) lane segment with v_T=30>15
                                        ], dtype=bool)

    static_action_space = StaticActionSpace(logger, filtering=filtering)

    action_specs = static_action_space.specify_goals(follow_lane_recipes,
                                                     behavioral_grid_state_with_segments_limits)

    action_spec_filter = ActionSpecFiltering(filters=[FilterSpecIfNone(), FilterForLaneSpeedLimits()], logger=logger)

    filter_results = action_spec_filter.filter_action_specs(action_specs,
                                                            behavioral_grid_state_with_segments_limits)

    np.testing.assert_array_equal(filter_results, expected_filter_results)

def test_filter_laneSpeedLimits_filtersSpecsViolatingLaneSpeedLimitsWhenSlowing_filterResultsMatchExpected(
        behavioral_grid_state_with_segments_limits,
        follow_lane_recipes: List[StaticActionRecipe]):

    logger = AV_Logger.get_logger()
    # The scene_static that is being used, is in accordance to whatever happens in behavioral_grid_state_
    # with_segments_limits fixture
    scene_static_with_limits = SceneStaticModel.get_instance().get_scene_static()
    # The following are 4 consecutive lane segments with varying speed limits (ego starts at the end of [0])
    # These are the s-values that correspond to lane transitions on the GFF:
    # [0.0, 100.84134201631973, 220.48438762415998, 343.9575891327402, 466.0989153990629]
    scene_static_with_limits.s_Data.s_SceneStaticBase.as_scene_lane_segments[0].e_v_nominal_speed = 4
    scene_static_with_limits.s_Data.s_SceneStaticBase.as_scene_lane_segments[3].e_v_nominal_speed = 4
    scene_static_with_limits.s_Data.s_SceneStaticBase.as_scene_lane_segments[6].e_v_nominal_speed = 4
    scene_static_with_limits.s_Data.s_SceneStaticBase.as_scene_lane_segments[9].e_v_nominal_speed = 4
    SceneStaticModel.get_instance().set_scene_static(scene_static_with_limits)

    filtering = RecipeFiltering(filters=[], logger=logger)
    # note: first lane segment speed limit is almost irrelevant because we start at the end of this segment
    expected_filter_results = np.array([True, True, True,      # v_T=0 (Calm, Standard, Aggressive)  - All Pass (not arriving at [9])
                                        False, False, False,   # v_T=6 - Fail  <<-- This was fixed by checking the final velocity is met
                                        False, False, False,   # v_T=12 - Fail
                                        False, False, False,   # v_T=18 - Fail
                                        False, False, False,   # v_T=24 - Fail
                                        False, False, False    # v_T=30 - Fail
                                        ], dtype=bool)

    static_action_space = StaticActionSpace(logger, filtering=filtering)

    action_specs = static_action_space.specify_goals(follow_lane_recipes,
                                                     behavioral_grid_state_with_segments_limits)

    action_spec_filter = ActionSpecFiltering(filters=[FilterSpecIfNone(), FilterForLaneSpeedLimits()], logger=logger)

    filter_results = action_spec_filter.filter_action_specs(action_specs,
                                                            behavioral_grid_state_with_segments_limits)

    np.testing.assert_array_equal(filter_results, expected_filter_results)


def test_filter_filterForSLimit_dontFilterValidAction(
        behavioral_grid_state_with_objects_for_filtering_too_aggressive,
        follow_vehicle_recipes_towards_front_cells: List[DynamicActionRecipe]):
    """
    State leads to two dynamic actions: {a0=0,v0=10,sT=53.5,vT=30} (SAME_LANE), {a0=0,v0=10,sT=53.5,vT=20} (RIGHT_LANE).
    The action on the right lane (with slow dynamic object) ends inside RIGHT_LANE Frenet frame and is not filtered.
    All ground truths checked with desmos - https://www.desmos.com/calculator/exizg3iuhs
    """
    logger = AV_Logger.get_logger()
    predictor = RoadFollowingPredictor(logger)

    filtering = RecipeFiltering(filters=[FilterRecipeIfNone()], logger=logger)

    actions_with_vehicle = follow_vehicle_recipes_towards_front_cells[:3]

    expected_filter_results = np.array([False, False, True], dtype=bool)
    dynamic_action_space = DynamicActionSpace(logger, predictor, filtering=filtering)

    action_specs_with_vehicle = dynamic_action_space.specify_goals(actions_with_vehicle,
                                                                   behavioral_grid_state_with_objects_for_filtering_too_aggressive)

    action_spec_filter = ActionSpecFiltering(filters=[FilterSpecIfNone(), FilterForSLimit()], logger=logger)

    filter_results = action_spec_filter.filter_action_specs(action_specs_with_vehicle,
                                                            behavioral_grid_state_with_objects_for_filtering_too_aggressive)

    np.testing.assert_array_equal(filter_results, expected_filter_results)


def test_filter_filterForSLimit_filterTooLongAction(
        behavioral_grid_state_with_objects_for_filtering_too_aggressive,
        follow_vehicle_recipes_towards_front_cells: List[DynamicActionRecipe]):
    """
    State leads to two dynamic actions: {a0=0,v0=10,sT=53.5,vT=30} (SAME_LANE), {a0=0,v0=10,sT=53.5,vT=20} (RIGHT_LANE).
    The action on the same lane ends beyond SAME_LANE Frenet frame and therefore is filtered.
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

    action_spec_filter = ActionSpecFiltering(filters=[FilterSpecIfNone(), FilterForSLimit()], logger=logger)

    filter_results = action_spec_filter.filter_action_specs(action_specs_with_vehicle,
                                                            behavioral_grid_state_with_objects_for_filtering_too_aggressive)

    np.testing.assert_array_equal(filter_results, expected_filter_results)
