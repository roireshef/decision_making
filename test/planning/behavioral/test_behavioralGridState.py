from decision_making.src.global_constants import MAX_BACKWARD_HORIZON, MAX_FORWARD_HORIZON
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import RelativeLongitudinalPosition
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GFFType
from rte.python.logger.AV_logger import AV_Logger
from unittest.mock import patch

from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.planning.behavioral.data_objects import RelativeLane
from decision_making.src.planning.types import FP_SX, FP_DX, FS_SX, FS_DX
from decision_making.src.utils.map_utils import MapUtils

import numpy as np

from decision_making.test.planning.behavioral.behavioral_state_fixtures import behavioral_grid_state, \
    state_with_surrounding_objects, state_with_surrounding_objects_and_off_map_objects, route_plan_20_30, \
    state_with_left_lane_ending, state_with_right_lane_ending, state_with_same_lane_ending_no_left_lane, \
    state_with_same_lane_ending_no_right_lane, state_with_lane_split_on_right, state_with_lane_split_on_left, \
    state_with_lane_split_on_left_and_right, state_with_lane_split_on_right_ending, route_plan_lane_split_on_right_ends, \
    state_with_lane_split_on_left_ending, route_plan_lane_split_on_left_ends, state_with_lane_split_on_left_and_right_ending, \
    route_plan_lane_splits_on_left_and_right_end, route_plan_lane_splits_on_left_and_right_left_first, \
    state_with_lane_split_on_left_and_right_left_first, route_plan_lane_splits_on_left_and_right_right_first, \
    state_with_lane_split_on_left_and_right_right_first, state_with_object_after_merge, state_with_objects_around_3to1_merge, \
    behavioral_grid_state_with_objects_for_filtering_too_aggressive, state_with_objects_for_filtering_too_aggressive, \
    route_plan_20_30, create_route_plan_msg, route_plan_lane_splits_on_left_and_right_left_first, \
    route_plan_lane_splits_on_left_and_right_right_first

from decision_making.test.planning.custom_fixtures import route_plan_1_2, route_plan_1_2_3, route_plan_left_lane_ends, route_plan_right_lane_ends, \
    route_plan_lane_split_on_right, route_plan_lane_split_on_left, route_plan_lane_split_on_left_and_right

from decision_making.test.messages.scene_static_fixture import scene_static_pg_split, right_lane_split_scene_static, \
    left_right_lane_split_scene_static, scene_static_short_testable, scene_static_left_lane_ends, scene_static_right_lane_ends, \
    left_lane_split_scene_static, scene_static_lane_split_on_left_ends, scene_static_lane_splits_on_left_and_right_left_first, \
    scene_static_lane_splits_on_left_and_right_right_first, scene_static_short_testable

SMALL_DISTANCE_ERROR = 0.01

def test_createFromState_objectAfterMerge_objectAssignedToAllGffs(state_with_object_after_merge, route_plan_1_2):
    behavioral_state = BehavioralGridState.create_from_state(state_with_object_after_merge, route_plan_1_2, None)
    # check to see if the road occupancy grid contains the actor for both same_lane and right_lane
    occupancy_grid = behavioral_state.road_occupancy_grid
    assert len(occupancy_grid[(RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT)]) > 0
    assert occupancy_grid[(RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT)][0].dynamic_object.obj_id == 1
    assert len(occupancy_grid[(RelativeLane.RIGHT_LANE, RelativeLongitudinalPosition.FRONT)]) > 0
    assert occupancy_grid[(RelativeLane.RIGHT_LANE, RelativeLongitudinalPosition.FRONT)][0].dynamic_object.obj_id == 1

def test_createFromState_objectsAroundMerge_objectsCorrectlyAssigned(state_with_objects_around_3to1_merge, route_plan_1_2):
    behavioral_state = BehavioralGridState.create_from_state(state_with_objects_around_3to1_merge, route_plan_1_2, None)

    occupancy_grid = behavioral_state.road_occupancy_grid
    assert len(occupancy_grid[(RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT)]) > 0
    front_same_grid_ids = [item.dynamic_object.obj_id for item in occupancy_grid[(RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT)]]
    front_left_grid_ids = [item.dynamic_object.obj_id for item in occupancy_grid[(RelativeLane.LEFT_LANE, RelativeLongitudinalPosition.FRONT)]]
    front_right_grid_ids = [item.dynamic_object.obj_id for item in occupancy_grid[(RelativeLane.RIGHT_LANE, RelativeLongitudinalPosition.FRONT)]]
    assert np.array_equal(np.sort(front_same_grid_ids), [1, 2, 3])
    assert np.array_equal(np.sort(front_left_grid_ids), [1, 2, 4])
    assert np.array_equal(np.sort(front_right_grid_ids), [1, 2, 5])


def test_createFromState_8objectsAroundEgo_correctGridSize(state_with_surrounding_objects, route_plan_20_30):
    """
    validate that 8 objects around ego create 8 grid cells in the behavioral state in multi-road map
    (a cell is created only if it contains at least one object)
    """
    logger = AV_Logger.get_logger()

    behavioral_state = BehavioralGridState.create_from_state(state_with_surrounding_objects, route_plan_20_30, logger)

    assert len(behavioral_state.road_occupancy_grid) == len(state_with_surrounding_objects.dynamic_objects)


def test_createFromState_eightObjectsAroundEgo_IgnoreThreeOffMapObjects(state_with_surrounding_objects_and_off_map_objects,
                                                                        route_plan_20_30):
    """
    Off map objects are located at ego's right lane.
    validate that 8 objects around ego create 5 grid cells in the behavioral state in multi-road map, while ignoring 3
    off map objects.
    (a cell is created only if it contains at least one on-map object, off map objects are marked with an off-map flag)
    """
    logger = AV_Logger.get_logger()
    behavioral_state = BehavioralGridState.create_from_state(
        state_with_surrounding_objects_and_off_map_objects, route_plan_20_30, logger)
    on_map_objects = [obj for obj in state_with_surrounding_objects_and_off_map_objects.dynamic_objects
                      if not obj.off_map]
    assert len(behavioral_state.road_occupancy_grid) == len(on_map_objects)

    for rel_lane, rel_lon in behavioral_state.road_occupancy_grid:
        assert rel_lane != RelativeLane.RIGHT_LANE
        dynamic_objects_on_grid = behavioral_state.road_occupancy_grid[(rel_lane, rel_lon)]
        assert np.all([not obj.dynamic_object.off_map for obj in dynamic_objects_on_grid])


def test_createFromState_leftLaneEnds_partialGffOnLeft(state_with_left_lane_ending, route_plan_left_lane_ends):
    """
    Host is in middle lane of three-lane road, and the left lane ends ahead. The left GFF should be a partial, and the other two should
    be normal.
    """
    behavioral_grid_state = BehavioralGridState.create_from_state(state_with_left_lane_ending, route_plan_left_lane_ends, None)
    gffs = behavioral_grid_state.extended_lane_frames

    # Check GFF Types
    assert gffs[RelativeLane.LEFT_LANE].gff_type == GFFType.Partial
    assert gffs[RelativeLane.SAME_LANE].gff_type == GFFType.Normal
    assert gffs[RelativeLane.RIGHT_LANE].gff_type == GFFType.Normal


def test_createFromState_rightLaneEnds_partialGffOnRight(state_with_right_lane_ending, route_plan_right_lane_ends):
    """
    Host is in middle lane of three-lane road, and the right lane ends ahead. The right GFF should be a partial, and the other two should
    be normal.
    """
    behavioral_grid_state = BehavioralGridState.create_from_state(state_with_right_lane_ending, route_plan_right_lane_ends, None)
    gffs = behavioral_grid_state.extended_lane_frames

    # Check GFF Types
    assert gffs[RelativeLane.LEFT_LANE].gff_type == GFFType.Normal
    assert gffs[RelativeLane.SAME_LANE].gff_type == GFFType.Normal
    assert gffs[RelativeLane.RIGHT_LANE].gff_type == GFFType.Partial


def test_createFromState_laneEndsNoLeftLane_partialGffInLaneNoLeftLane(state_with_same_lane_ending_no_left_lane, route_plan_left_lane_ends):
    """
    Host is on three-lane road, and is in the furthest left lane that ends ahead. A left GFF should not be created, the same lane GFF
    should be partial, and the right lane GFF should be normal.
    """
    behavioral_grid_state = BehavioralGridState.create_from_state(state_with_same_lane_ending_no_left_lane, route_plan_left_lane_ends, None)
    gffs = behavioral_grid_state.extended_lane_frames

    # Check GFF Types
    assert gffs[RelativeLane.SAME_LANE].gff_type == GFFType.Partial
    assert gffs[RelativeLane.RIGHT_LANE].gff_type == GFFType.Normal

    # Check that left GFF does not exist
    assert RelativeLane.LEFT_LANE not in gffs


def test_createFromState_laneEndsNoRightLane_partialGffInLaneNoRightLane(state_with_same_lane_ending_no_right_lane,
                                                                         route_plan_right_lane_ends):
    """
    Host is on three-lane road, and is in the furthest right lane that ends ahead. A right GFF should not be created, the same lane GFF
    should be partial, and the left lane GFF should be normal.
    """
    behavioral_grid_state = BehavioralGridState.create_from_state(state_with_same_lane_ending_no_right_lane, route_plan_right_lane_ends, None)
    gffs = behavioral_grid_state.extended_lane_frames

    # Check GFF Types
    assert gffs[RelativeLane.LEFT_LANE].gff_type == GFFType.Normal
    assert gffs[RelativeLane.SAME_LANE].gff_type == GFFType.Partial

    # Check that right GFF does not exist
    assert RelativeLane.RIGHT_LANE not in gffs


def test_createFromState_laneSplitOnRight_augmentedGffOnRight(state_with_lane_split_on_right, route_plan_lane_split_on_right):
    """
    Host is in right lane of two-lane road, and a lane split on the right is ahead. The right GFF should be augmented, and the other two
    should be normal.
    """
    behavioral_grid_state = BehavioralGridState.create_from_state(state_with_lane_split_on_right, route_plan_lane_split_on_right, None)
    gffs = behavioral_grid_state.extended_lane_frames

    # Check GFF Types
    assert gffs[RelativeLane.LEFT_LANE].gff_type == GFFType.Normal
    assert gffs[RelativeLane.SAME_LANE].gff_type == GFFType.Normal
    assert gffs[RelativeLane.RIGHT_LANE].gff_type == GFFType.Augmented


def test_createFromState_laneSplitOnLeft_augmentedGffOnLeft(state_with_lane_split_on_left, route_plan_lane_split_on_left):
    """
    Host is in left lane of two-lane road, and a lane split on the left is ahead. The left GFF should be augmented, and the other two
    should be normal.
    """
    behavioral_grid_state = BehavioralGridState.create_from_state(state_with_lane_split_on_left, route_plan_lane_split_on_left, None)
    gffs = behavioral_grid_state.extended_lane_frames

    # Check GFF Types
    assert gffs[RelativeLane.LEFT_LANE].gff_type == GFFType.Augmented
    assert gffs[RelativeLane.SAME_LANE].gff_type == GFFType.Normal
    assert gffs[RelativeLane.RIGHT_LANE].gff_type == GFFType.Normal


def test_createFromState_laneSplitOnLeftAndRight_augmentedGffOnLeftAndRight(state_with_lane_split_on_left_and_right,
                                                                            route_plan_lane_split_on_left_and_right):
    """
    Host is on one-lane road, and lane splits on the left and right are ahead. The left and right GFFs should be augmented, and the same
    lane GFF should be normal.
    """
    behavioral_grid_state = BehavioralGridState.create_from_state(state_with_lane_split_on_left_and_right,
                                                                  route_plan_lane_split_on_left_and_right, None)
    gffs = behavioral_grid_state.extended_lane_frames

    # Check GFF Types
    assert gffs[RelativeLane.LEFT_LANE].gff_type == GFFType.Augmented
    assert gffs[RelativeLane.SAME_LANE].gff_type == GFFType.Normal
    assert gffs[RelativeLane.RIGHT_LANE].gff_type == GFFType.Augmented


def test_createFromState_laneSplitOnRight_augmentedPartialGffOnRight(state_with_lane_split_on_right_ending,
                                                                     route_plan_lane_split_on_right_ends):
    """
    Host is in right lane of two-lane road, a lane split on the right is ahead, and the new lane ends shortly thereafter. The right GFF
    should be augmented partial, and the other two should be normal.
    """
    behavioral_grid_state = BehavioralGridState.create_from_state(state_with_lane_split_on_right_ending,
                                                                  route_plan_lane_split_on_right_ends, None)
    gffs = behavioral_grid_state.extended_lane_frames

    # Check GFF Types
    assert gffs[RelativeLane.LEFT_LANE].gff_type == GFFType.Normal
    assert gffs[RelativeLane.SAME_LANE].gff_type == GFFType.Normal
    assert gffs[RelativeLane.RIGHT_LANE].gff_type == GFFType.AugmentedPartial


def test_createFromState_laneSplitOnLeft_augmentedPartialGffOnLeft(state_with_lane_split_on_left_ending,
                                                                   route_plan_lane_split_on_left_ends):
    """
    Host is in left lane of two-lane road, a lane split on the left is ahead, and the new lane ends shortly thereafter. The left GFF
    should be augmented partial, and the other two should be normal.
    """
    behavioral_grid_state = BehavioralGridState.create_from_state(state_with_lane_split_on_left_ending,
                                                                  route_plan_lane_split_on_left_ends, None)
    gffs = behavioral_grid_state.extended_lane_frames

    # Check GFF Types
    assert gffs[RelativeLane.LEFT_LANE].gff_type == GFFType.AugmentedPartial
    assert gffs[RelativeLane.SAME_LANE].gff_type == GFFType.Normal
    assert gffs[RelativeLane.RIGHT_LANE].gff_type == GFFType.Normal


def test_createFromState_laneSplitOnLeftAndRight_augmentedPartialGffOnLeftAndRight(state_with_lane_split_on_left_and_right_ending,
                                                                                   route_plan_lane_splits_on_left_and_right_end):
    """
    Host is on one-lane road, lane splits on the left and right are ahead, and the new lanes end shortly thereafter. The left and right GFFs
    should be augmented partial, and the same lane GFF should be normal.
    """
    behavioral_grid_state = BehavioralGridState.create_from_state(state_with_lane_split_on_left_and_right_ending,
                                                                  route_plan_lane_splits_on_left_and_right_end, None)
    gffs = behavioral_grid_state.extended_lane_frames

    # Check GFF Types
    assert gffs[RelativeLane.LEFT_LANE].gff_type == GFFType.AugmentedPartial
    assert gffs[RelativeLane.SAME_LANE].gff_type == GFFType.Normal
    assert gffs[RelativeLane.RIGHT_LANE].gff_type == GFFType.AugmentedPartial


def test_createFromState_laneSplitOnLeftAndRightLeftFirst_augmentedGffOnLeftAndRightOffset(state_with_lane_split_on_left_and_right_left_first,
                                                                                           route_plan_lane_splits_on_left_and_right_left_first):
    """
    Host is on one-lane road and lane splits on the left and right are ahead.
    The left split comes before the right split.
    The left and right GFFs should be augmented, and the same lane GFF should be normal.
    """
    behavioral_grid_state = BehavioralGridState.create_from_state(state_with_lane_split_on_left_and_right_left_first,
                                                                  route_plan_lane_splits_on_left_and_right_left_first, None)
    gffs = behavioral_grid_state.extended_lane_frames

    # Check GFF Types
    assert gffs[RelativeLane.LEFT_LANE].gff_type == GFFType.Augmented
    assert gffs[RelativeLane.SAME_LANE].gff_type == GFFType.Normal
    assert gffs[RelativeLane.RIGHT_LANE].gff_type == GFFType.Augmented


def test_createFromState_laneSplitOnLeftAndRightRightFirst_augmentedGffOnLeftAndRightOffset(state_with_lane_split_on_left_and_right_right_first,
                                                                                            route_plan_lane_splits_on_left_and_right_right_first):
    """
    Host is on one-lane road and lane splits on the left and right are ahead.
    The left split comes after the right split.
    The left and right GFFs should be augmented, and the same lane GFF should be normal.
    """
    behavioral_grid_state = BehavioralGridState.create_from_state(state_with_lane_split_on_left_and_right_right_first,
                                                                  route_plan_lane_splits_on_left_and_right_right_first, None)
    gffs = behavioral_grid_state.extended_lane_frames

    # Check GFF Types
    assert gffs[RelativeLane.LEFT_LANE].gff_type == GFFType.Augmented
    assert gffs[RelativeLane.SAME_LANE].gff_type == GFFType.Normal
    assert gffs[RelativeLane.RIGHT_LANE].gff_type == GFFType.Augmented

def test_calculateLongitudinalDifferences_8objectsAroundEgo_accurate(state_with_surrounding_objects, behavioral_grid_state):
    """
    validate that 8 objects around ego have accurate longitudinal distances from ego in multi-road map
    """
    target_map_states = [obj.map_state for obj in state_with_surrounding_objects.dynamic_objects]

    longitudinal_distances = behavioral_grid_state.calculate_longitudinal_differences(target_map_states)

    for i, map_state in enumerate(target_map_states):
        ego_ordinal = MapUtils.get_lane_ordinal(behavioral_grid_state.ego_state.map_state.lane_id)
        target_ordinal = MapUtils.get_lane_ordinal(map_state.lane_id)
        rel_lane = RelativeLane(target_ordinal - ego_ordinal)
        target_gff_fstate = behavioral_grid_state.extended_lane_frames[rel_lane].convert_from_segment_state(
            map_state.lane_fstate, map_state.lane_id)
        assert longitudinal_distances[i] == target_gff_fstate[FS_SX] - behavioral_grid_state.projected_ego_fstates[rel_lane][FS_SX]


def test_getGeneralizedFrenetFrameByCost_onEndingLane_PartialGFFCreated(scene_static_left_lane_ends, route_plan_1_2):
    """
    Make sure a partial GFF is created when the left lane suddenly ends
    :param scene_static_left_lane_ends:
    :return:
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_left_lane_ends)

    starting_lon = 800
    starting_lane = 12

    del route_plan_1_2.s_Data.as_route_plan_lane_segments[1][2]
    route_plan_1_2.s_Data.a_Cnt_num_lane_segments[1] -= 1

    gff_dict = BehavioralGridState.get_generalized_frenet_frames_by_cost(starting_lane, starting_lon, route_plan_1_2)
    # check partial SAME_LANE
    assert np.array_equal(gff_dict[RelativeLane.SAME_LANE].segment_ids, [12])
    assert gff_dict[RelativeLane.SAME_LANE].gff_type == GFFType.Partial


def test_getGeneralizedFrenetFrameByCost_onFullLane_NormalGFFCreated(scene_static_right_lane_ends, route_plan_1_2):
    SceneStaticModel.get_instance().set_scene_static(scene_static_right_lane_ends)

    starting_lon = 800
    starting_lane = 11

    gff_dict = BehavioralGridState.get_generalized_frenet_frames_by_cost(starting_lane, starting_lon, route_plan_1_2)
    assert np.array_equal(gff_dict[RelativeLane.SAME_LANE].segment_ids, [11,21])
    assert gff_dict[RelativeLane.SAME_LANE].gff_type == GFFType.Normal


def test_getGeneralizedFrenetFrameByCost_LeftSplitAugmentedGFFCreated(left_lane_split_scene_static, route_plan_1_2):
    SceneStaticModel.get_instance().set_scene_static(left_lane_split_scene_static)

    starting_lon = 700.
    starting_lane = 11
    can_augment = {RelativeLane.LEFT_LANE: True, RelativeLane.RIGHT_LANE: False}

    gff_dict = BehavioralGridState.get_generalized_frenet_frames_by_cost(starting_lane, starting_lon, route_plan_1_2, can_augment = can_augment)

    # check same_lane
    assert gff_dict[RelativeLane.SAME_LANE].gff_type == GFFType.Normal
    assert np.array_equal(gff_dict[RelativeLane.SAME_LANE].segment_ids, [11, 21])
    # check augmented right lane
    assert gff_dict[RelativeLane.LEFT_LANE].gff_type == GFFType.Augmented
    assert np.array_equal(gff_dict[RelativeLane.LEFT_LANE].segment_ids, [11, 22])


def test_getGeneralizedFrenetFrameByCost_RightSplitAugmentedGFFCreated(right_lane_split_scene_static, route_plan_1_2):
    SceneStaticModel.get_instance().set_scene_static(right_lane_split_scene_static)

    starting_lon = 700.
    starting_lane = 11
    can_augment = {RelativeLane.LEFT_LANE: False, RelativeLane.RIGHT_LANE: True}

    gff_dict = BehavioralGridState.get_generalized_frenet_frames_by_cost(starting_lane, starting_lon, route_plan_1_2, can_augment=can_augment)

    # check same_lane
    assert gff_dict[RelativeLane.SAME_LANE].gff_type == GFFType.Normal
    assert np.array_equal(gff_dict[RelativeLane.SAME_LANE].segment_ids, [11, 21])
    # check augmented right lane
    assert gff_dict[RelativeLane.RIGHT_LANE].gff_type == GFFType.Augmented
    assert np.array_equal(gff_dict[RelativeLane.RIGHT_LANE].segment_ids, [11, 20])


def test_getGeneralizedFrenetFrameByCost_LeftRightSplitAugmentedGFFsCreated(left_right_lane_split_scene_static, route_plan_1_2):
    SceneStaticModel.get_instance().set_scene_static(left_right_lane_split_scene_static)
    can_augment = {RelativeLane.LEFT_LANE: True, RelativeLane.RIGHT_LANE: True}

    # Modify the route plan
    # In order to match the scene static data, the left and right lane in the first road segment needs to be deleted since
    # it does not exist in left_right_lane_split_scene_static.
    del route_plan_1_2.s_Data.as_route_plan_lane_segments[0][0]
    # delete index [0][1] instead of [0][2] since the first delete shifts all the indicies
    del route_plan_1_2.s_Data.as_route_plan_lane_segments[0][1]
    route_plan_1_2.s_Data.a_Cnt_num_lane_segments[0] = 1

    gff_dict = BehavioralGridState.get_generalized_frenet_frames_by_cost(11, 600, route_plan_1_2, can_augment=can_augment)

    assert gff_dict[RelativeLane.LEFT_LANE].gff_type == GFFType.Augmented
    assert gff_dict[RelativeLane.RIGHT_LANE].gff_type == GFFType.Augmented
    assert gff_dict[RelativeLane.SAME_LANE].gff_type == GFFType.Normal
    assert gff_dict[RelativeLane.LEFT_LANE].has_segment_id(22)
    assert gff_dict[RelativeLane.RIGHT_LANE].has_segment_id(20)
    assert gff_dict[RelativeLane.SAME_LANE].has_segment_id(21)


def test_getGeneralizedFrenetFrameByCost_CanAugmentButNoSplit_NoAugmentedCreated(scene_static_short_testable, route_plan_1_2):
    SceneStaticModel.get_instance().set_scene_static(scene_static_short_testable)
    starting_lon = 700.
    starting_lane = 11
    can_augment = {RelativeLane.LEFT_LANE: True, RelativeLane.RIGHT_LANE: True}

    gff_dict = BehavioralGridState.get_generalized_frenet_frames_by_cost(starting_lane, starting_lon, route_plan_1_2, can_augment=can_augment)

    # check same_lane
    assert gff_dict[RelativeLane.SAME_LANE].gff_type == GFFType.Normal
    assert np.array_equal(gff_dict[RelativeLane.SAME_LANE].segment_ids, [11, 21])
    assert RelativeLane.LEFT_LANE not in gff_dict
    assert RelativeLane.RIGHT_LANE not in gff_dict

@patch('decision_making.src.planning.behavioral.behavioral_grid_state.MAX_FORWARD_HORIZON', 400)
def test_getGeneralizedFrenetFrameByCost_OffsetSplitsLeftFirst_BothAugmentedCreated(scene_static_lane_splits_on_left_and_right_left_first,
                                                                                  route_plan_lane_splits_on_left_and_right_left_first):
    SceneStaticModel.get_instance().set_scene_static(scene_static_lane_splits_on_left_and_right_left_first)
    starting_lon = 10.
    starting_lane = 211
    can_augment = {RelativeLane.LEFT_LANE: True, RelativeLane.RIGHT_LANE: True}

    gff_dict = BehavioralGridState.get_generalized_frenet_frames_by_cost(starting_lane, starting_lon, route_plan_lane_splits_on_left_and_right_left_first,
                                                                         can_augment=can_augment)

    assert gff_dict[RelativeLane.SAME_LANE].gff_type == GFFType.Normal
    assert gff_dict[RelativeLane.LEFT_LANE].gff_type == GFFType.Augmented
    assert gff_dict[RelativeLane.RIGHT_LANE].gff_type == GFFType.Augmented
    assert np.array_equal(gff_dict[RelativeLane.SAME_LANE].segment_ids, [201, 211, 221, 231, 241])
    assert np.array_equal(gff_dict[RelativeLane.LEFT_LANE].segment_ids, [201, 211, 222, 232, 242])
    assert np.array_equal(gff_dict[RelativeLane.RIGHT_LANE].segment_ids, [201, 211, 221, 230, 240])

@patch('decision_making.src.planning.behavioral.behavioral_grid_state.MAX_FORWARD_HORIZON', 400)
def test_getGeneralizedFrenetFrameByCost_OffsetSplitsRightFirst_BothAugmentedCreated(scene_static_lane_splits_on_left_and_right_right_first,
                                                                                   route_plan_lane_splits_on_left_and_right_right_first):
    SceneStaticModel.get_instance().set_scene_static(scene_static_lane_splits_on_left_and_right_right_first)
    starting_lon = 10.
    starting_lane = 211
    can_augment = {RelativeLane.LEFT_LANE: True, RelativeLane.RIGHT_LANE: True}

    gff_dict = BehavioralGridState.get_generalized_frenet_frames_by_cost(starting_lane, starting_lon, route_plan_lane_splits_on_left_and_right_right_first,
                                                                         can_augment=can_augment)

    assert gff_dict[RelativeLane.SAME_LANE].gff_type == GFFType.Normal
    assert gff_dict[RelativeLane.LEFT_LANE].gff_type == GFFType.Augmented
    assert gff_dict[RelativeLane.RIGHT_LANE].gff_type == GFFType.Augmented
    assert np.array_equal(gff_dict[RelativeLane.SAME_LANE].segment_ids, [201, 211, 221, 231, 241])
    assert np.array_equal(gff_dict[RelativeLane.LEFT_LANE].segment_ids, [201, 211, 221, 232, 242])
    assert np.array_equal(gff_dict[RelativeLane.RIGHT_LANE].segment_ids, [201, 211, 220, 230, 240])


def test_getGeneralizedFrenetFrameByCost_frenetStartsBehindAndEndsAheadOfCurrentLane_accurateFrameStartAndLength(
        scene_static_pg_split, route_plan_20_30):
    """
    test method get_generalized_frenet_frame_by_cost:
        the frame starts and ends on arbitrary points.
    verify that final length, offset of GFF and conversion of an arbitrary point are accurate
    """

    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    road_ids = MapUtils.get_road_segment_ids()
    current_road_idx = 3
    current_ordinal = 1
    station = 50.0
    arbitrary_fpoint = np.array([450., 1.])

    lane_ids = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[current_road_idx])
    lane_id = lane_ids[current_ordinal]
    gff = BehavioralGridState.get_generalized_frenet_frames_by_cost(lane_id, station, route_plan_20_30)[RelativeLane.SAME_LANE]

    # validate the length of the obtained frenet frame
    assert abs(gff.s_max - (MAX_BACKWARD_HORIZON + MAX_FORWARD_HORIZON)) < SMALL_DISTANCE_ERROR
    # calculate cartesian state of the origin of lane_id using GFF and using original frenet of lane_id and compare them
    gff_cpoint = gff.fpoint_to_cpoint(np.array([MAX_BACKWARD_HORIZON - station, 0]))
    ff_cpoint = MapUtils.get_lane_frenet_frame(lane_id).fpoint_to_cpoint(np.array([0, 0]))
    assert np.linalg.norm(gff_cpoint - ff_cpoint) < SMALL_DISTANCE_ERROR

    # calculate cartesian state of some point using GFF and using original frenet (from the map) and compare them
    gff_cpoint = gff.fpoint_to_cpoint(arbitrary_fpoint)
    segment_id, segment_fstate = gff.convert_to_segment_state(np.array([arbitrary_fpoint[FP_SX], 0, 0,
                                                                        arbitrary_fpoint[FP_DX], 0, 0]))
    ff_cpoint = MapUtils.get_lane_frenet_frame(segment_id).fpoint_to_cpoint(segment_fstate[[FS_SX, FS_DX]])
    assert np.linalg.norm(gff_cpoint - ff_cpoint) < SMALL_DISTANCE_ERROR


@patch('decision_making.src.planning.behavioral.behavioral_grid_state.MAX_FORWARD_HORIZON', 900)
def test_getGeneralizedFrenet_AugmentedPartialCreatedWhenSplitEnds(left_right_lane_split_scene_static, route_plan_1_2_3):
    """
    Make sure that partial/augmentedPartial GFFS are created when the forward horizon is set to be very far ahead
    :param left_right_lane_split_scene_static:
    :param route_plan_1_2_3:
    :return:
    """
    SceneStaticModel.get_instance().set_scene_static(left_right_lane_split_scene_static)
    can_augment = {RelativeLane.LEFT_LANE: True, RelativeLane.RIGHT_LANE: True}

    # Modify the route plan
    # In order to match the scene static data, the left and right lane in the first road segment needs to be deleted since
    # it does not exist in left_right_lane_split_scene_static.
    del route_plan_1_2_3.s_Data.as_route_plan_lane_segments[0][0]
    # delete index [0][1] instead of [0][2] since the first delete shifts all the indicies
    del route_plan_1_2_3.s_Data.as_route_plan_lane_segments[0][1]
    route_plan_1_2_3.s_Data.a_Cnt_num_lane_segments[0] = 1

    gff_dict = BehavioralGridState.get_generalized_frenet_frames_by_cost(11, 900, route_plan_1_2_3, can_augment=can_augment)

    assert gff_dict[RelativeLane.LEFT_LANE].gff_type == GFFType.AugmentedPartial
    assert gff_dict[RelativeLane.RIGHT_LANE].gff_type == GFFType.AugmentedPartial
    assert gff_dict[RelativeLane.SAME_LANE].gff_type == GFFType.Partial
    assert gff_dict[RelativeLane.LEFT_LANE].has_segment_id(22)
    assert gff_dict[RelativeLane.RIGHT_LANE].has_segment_id(20)
    assert gff_dict[RelativeLane.SAME_LANE].has_segment_id(21)
