from decision_making.src.global_constants import MAX_BACKWARD_HORIZON, MAX_FORWARD_HORIZON
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import RelativeLongitudinalPosition
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GFFType
from rte.python.logger.AV_logger import AV_Logger
from unittest.mock import patch
from decision_making.src.planning.behavioral.data_objects import RelativeLane
from decision_making.src.planning.types import FP_SX, FP_DX, FS_SX, FS_DX
from decision_making.src.utils.map_utils import MapUtils
import numpy as np
from decision_making.src.messages.scene_static_message import SceneStatic
from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.state.state import DynamicObject, MapState, ObjectSize
from decision_making.src.messages.route_plan_message import RoutePlanLaneSegment
from decision_making.src.exceptions import NavigationPlanTooShort, UpstreamLaneNotFound

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
    route_plan_lane_splits_on_left_and_right_right_first, state_with_five_objects_on_oval_track, route_plan_for_oval_track_file

from decision_making.test.planning.custom_fixtures import route_plan_1_2, route_plan_1_2_3, route_plan_left_lane_ends, route_plan_right_lane_ends, \
    route_plan_lane_split_on_right, route_plan_lane_split_on_left, route_plan_lane_split_on_left_and_right

from decision_making.test.messages.scene_static_fixture import scene_static_pg_split, right_lane_split_scene_static, \
    left_right_lane_split_scene_static, scene_static_short_testable, scene_static_left_lane_ends, scene_static_right_lane_ends, \
    left_lane_split_scene_static, scene_static_lane_split_on_left_ends, scene_static_lane_splits_on_left_and_right_left_first, \
    scene_static_lane_splits_on_left_and_right_right_first, scene_static_oval_with_splits

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


def test_createProjectedObjects_laneSplit_carInOverlap(scene_static_oval_with_splits: SceneStatic):
    """
    Validate that projected object is correctly placed in overlapping lane
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_oval_with_splits)

    # Create other car in lane 21, which overlaps with lane 22
    dyn_obj = DynamicObject.create_from_map_state(obj_id=10, timestamp=5,
                                                  map_state=MapState(np.array([5,1,0,0,0,0]), 19670532),
                                                  size=ObjectSize(5, 2, 2), confidence=1, off_map=False)
    projected_dynamic_objects = BehavioralGridState._create_projected_objects([dyn_obj])

    assert projected_dynamic_objects[0].map_state.lane_id == 19670533


def test_createProjectedObjects_laneMerge_carInOverlap(scene_static_oval_with_splits: SceneStatic):
    """
    Validate that projected object is correctly placed in overlapping lane
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_oval_with_splits)

    overlaps = {}
    geo_lane_ids = [lane.e_i_lane_segment_id  for lane in scene_static_oval_with_splits.s_Data.s_SceneStaticGeometry.as_scene_lane_segments]
    for lane in scene_static_oval_with_splits.s_Data.s_SceneStaticBase.as_scene_lane_segments:
        if lane.e_Cnt_lane_overlap_count > 0 and lane.e_i_lane_segment_id in geo_lane_ids:
            for overlap in lane.as_lane_overlaps:
                overlaps[lane.e_i_lane_segment_id] = (
                overlap.e_i_other_lane_segment_id, overlap.e_e_lane_overlap_type)

    # Create other car in lane 21, which overlaps with lane 22
    dyn_obj = DynamicObject.create_from_map_state(obj_id=10, timestamp=5,
                                                  map_state=MapState(np.array([5, 1, 0, 0, 0, 0]), 58375684),
                                                  size=ObjectSize(5, 2, 2), confidence=1, off_map=False)
    projected_dynamic_objects = BehavioralGridState._create_projected_objects([dyn_obj])

    assert projected_dynamic_objects[0].map_state.lane_id == 58375685


def test_createProjectedObjects_laneSplit_carNotInOverlap(scene_static_short_testable: SceneStatic):
    """
    Validate that no mirror object is created if there is no overlap
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_short_testable)
    # Create other car in lane 21 which does NOT overlap with any other lane
    dyn_obj = DynamicObject.create_from_map_state(obj_id=10, timestamp=5, map_state=MapState(np.array([1,1,0,0,0,0]), 21),
                                                  size=ObjectSize(1,1,1), confidence=1, off_map=False)
    projected_dynamic_objects = BehavioralGridState._create_projected_objects([dyn_obj])

    assert not projected_dynamic_objects


@patch('decision_making.src.planning.behavioral.behavioral_grid_state.MAX_FORWARD_HORIZON', 20)
@patch('decision_making.src.planning.behavioral.behavioral_grid_state.MAX_BACKWARD_HORIZON', 0)
def test_filterIrrelevantDynamicObjects_fiveLanesOnOvalTrackWithObjects_IrrelevantObjectsFiltered(
        state_with_five_objects_on_oval_track, route_plan_for_oval_track_file):
    """
    Validate that irrelevant objects are filtered correctly
    """
    extended_lane_frames = BehavioralGridState._create_generalized_frenet_frames(
        state_with_five_objects_on_oval_track, route_plan_for_oval_track_file, None)

    relevant_objects, relevant_objects_relative_lanes = BehavioralGridState._filter_irrelevant_dynamic_objects(
        state_with_five_objects_on_oval_track.dynamic_objects, extended_lane_frames)

    relevant_object_lane_ids = [dynamic_object.map_state.lane_id for dynamic_object in relevant_objects]

    assert relevant_object_lane_ids == [19670530, 19670531, 19670532]
    assert relevant_objects_relative_lanes == [[RelativeLane.LEFT_LANE], [RelativeLane.SAME_LANE], [RelativeLane.RIGHT_LANE]]


def test_getGeneralizedFrenetFrames_onEndingLane_PartialGFFCreated(scene_static_left_lane_ends, route_plan_1_2):
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

    gff_dict = BehavioralGridState._get_generalized_frenet_frames(starting_lane, starting_lon, route_plan_1_2)
    # check partial SAME_LANE
    assert np.array_equal(gff_dict[RelativeLane.SAME_LANE].segment_ids, [12])
    assert gff_dict[RelativeLane.SAME_LANE].gff_type == GFFType.Partial


def test_getGeneralizedFrenetFrames_onFullLane_NormalGFFCreated(scene_static_right_lane_ends, route_plan_1_2):
    SceneStaticModel.get_instance().set_scene_static(scene_static_right_lane_ends)

    starting_lon = 800
    starting_lane = 11

    gff_dict = BehavioralGridState._get_generalized_frenet_frames(starting_lane, starting_lon, route_plan_1_2)
    assert np.array_equal(gff_dict[RelativeLane.SAME_LANE].segment_ids, [11,21])
    assert gff_dict[RelativeLane.SAME_LANE].gff_type == GFFType.Normal


def test_getGeneralizedFrenetFrames_LeftSplitAugmentedGFFCreated(left_lane_split_scene_static, route_plan_1_2):
    SceneStaticModel.get_instance().set_scene_static(left_lane_split_scene_static)

    starting_lon = 700.
    starting_lane = 11
    can_augment = [RelativeLane.LEFT_LANE]

    gff_dict = BehavioralGridState._get_generalized_frenet_frames(starting_lane, starting_lon, route_plan_1_2, can_augment = can_augment)

    # check same_lane
    assert gff_dict[RelativeLane.SAME_LANE].gff_type == GFFType.Normal
    assert np.array_equal(gff_dict[RelativeLane.SAME_LANE].segment_ids, [11, 21])
    # check augmented right lane
    assert gff_dict[RelativeLane.LEFT_LANE].gff_type == GFFType.Augmented
    assert np.array_equal(gff_dict[RelativeLane.LEFT_LANE].segment_ids, [11, 22])


def test_getGeneralizedFrenetFrames_RightSplitAugmentedGFFCreated(right_lane_split_scene_static, route_plan_1_2):
    SceneStaticModel.get_instance().set_scene_static(right_lane_split_scene_static)

    starting_lon = 700.
    starting_lane = 11
    can_augment = [RelativeLane.RIGHT_LANE]

    gff_dict = BehavioralGridState._get_generalized_frenet_frames(starting_lane, starting_lon, route_plan_1_2, can_augment=can_augment)

    # check same_lane
    assert gff_dict[RelativeLane.SAME_LANE].gff_type == GFFType.Normal
    assert np.array_equal(gff_dict[RelativeLane.SAME_LANE].segment_ids, [11, 21])
    # check augmented right lane
    assert gff_dict[RelativeLane.RIGHT_LANE].gff_type == GFFType.Augmented
    assert np.array_equal(gff_dict[RelativeLane.RIGHT_LANE].segment_ids, [11, 20])


def test_getGeneralizedFrenetFrames_LeftRightSplitAugmentedGFFsCreated(left_right_lane_split_scene_static, route_plan_1_2):
    SceneStaticModel.get_instance().set_scene_static(left_right_lane_split_scene_static)
    can_augment = [RelativeLane.LEFT_LANE, RelativeLane.RIGHT_LANE]

    # Modify the route plan
    # In order to match the scene static data, the left and right lane in the first road segment needs to be deleted since
    # it does not exist in left_right_lane_split_scene_static.
    del route_plan_1_2.s_Data.as_route_plan_lane_segments[0][0]
    # delete index [0][1] instead of [0][2] since the first delete shifts all the indicies
    del route_plan_1_2.s_Data.as_route_plan_lane_segments[0][1]
    route_plan_1_2.s_Data.a_Cnt_num_lane_segments[0] = 1

    gff_dict = BehavioralGridState._get_generalized_frenet_frames(11, 600, route_plan_1_2, can_augment=can_augment)

    assert gff_dict[RelativeLane.LEFT_LANE].gff_type == GFFType.Augmented
    assert gff_dict[RelativeLane.RIGHT_LANE].gff_type == GFFType.Augmented
    assert gff_dict[RelativeLane.SAME_LANE].gff_type == GFFType.Normal
    assert gff_dict[RelativeLane.LEFT_LANE].has_segment_id(22)
    assert gff_dict[RelativeLane.RIGHT_LANE].has_segment_id(20)
    assert gff_dict[RelativeLane.SAME_LANE].has_segment_id(21)


def test_getGeneralizedFrenetFrames_CanAugmentButNoSplit_NoAugmentedCreated(scene_static_short_testable, route_plan_1_2):
    SceneStaticModel.get_instance().set_scene_static(scene_static_short_testable)
    starting_lon = 700.
    starting_lane = 11
    can_augment = [RelativeLane.LEFT_LANE, RelativeLane.RIGHT_LANE]

    gff_dict = BehavioralGridState._get_generalized_frenet_frames(starting_lane, starting_lon, route_plan_1_2, can_augment=can_augment)

    # check same_lane
    assert gff_dict[RelativeLane.SAME_LANE].gff_type == GFFType.Normal
    assert np.array_equal(gff_dict[RelativeLane.SAME_LANE].segment_ids, [11, 21])
    assert RelativeLane.LEFT_LANE not in gff_dict
    assert RelativeLane.RIGHT_LANE not in gff_dict

@patch('decision_making.src.planning.behavioral.behavioral_grid_state.MAX_FORWARD_HORIZON', 400)
def test_getGeneralizedFrenetFrames_OffsetSplitsLeftFirst_BothAugmentedCreated(scene_static_lane_splits_on_left_and_right_left_first,
                                                                                  route_plan_lane_splits_on_left_and_right_left_first):
    SceneStaticModel.get_instance().set_scene_static(scene_static_lane_splits_on_left_and_right_left_first)
    starting_lon = 10.
    starting_lane = 211
    can_augment = [RelativeLane.LEFT_LANE, RelativeLane.RIGHT_LANE]

    gff_dict = BehavioralGridState._get_generalized_frenet_frames(starting_lane, starting_lon, route_plan_lane_splits_on_left_and_right_left_first,
                                                                  can_augment=can_augment)

    assert gff_dict[RelativeLane.SAME_LANE].gff_type == GFFType.Normal
    assert gff_dict[RelativeLane.LEFT_LANE].gff_type == GFFType.Augmented
    assert gff_dict[RelativeLane.RIGHT_LANE].gff_type == GFFType.Augmented
    assert np.array_equal(gff_dict[RelativeLane.SAME_LANE].segment_ids, [201, 211, 221, 231, 241])
    assert np.array_equal(gff_dict[RelativeLane.LEFT_LANE].segment_ids, [201, 211, 222, 232, 242])
    assert np.array_equal(gff_dict[RelativeLane.RIGHT_LANE].segment_ids, [201, 211, 221, 230, 240])

@patch('decision_making.src.planning.behavioral.behavioral_grid_state.MAX_FORWARD_HORIZON', 400)
def test_getGeneralizedFrenetFrames_OffsetSplitsRightFirst_BothAugmentedCreated(scene_static_lane_splits_on_left_and_right_right_first,
                                                                                   route_plan_lane_splits_on_left_and_right_right_first):
    SceneStaticModel.get_instance().set_scene_static(scene_static_lane_splits_on_left_and_right_right_first)
    starting_lon = 10.
    starting_lane = 211
    can_augment = [RelativeLane.LEFT_LANE, RelativeLane.RIGHT_LANE]

    gff_dict = BehavioralGridState._get_generalized_frenet_frames(starting_lane, starting_lon, route_plan_lane_splits_on_left_and_right_right_first,
                                                                  can_augment=can_augment)

    assert gff_dict[RelativeLane.SAME_LANE].gff_type == GFFType.Normal
    assert gff_dict[RelativeLane.LEFT_LANE].gff_type == GFFType.Augmented
    assert gff_dict[RelativeLane.RIGHT_LANE].gff_type == GFFType.Augmented
    assert np.array_equal(gff_dict[RelativeLane.SAME_LANE].segment_ids, [201, 211, 221, 231, 241])
    assert np.array_equal(gff_dict[RelativeLane.LEFT_LANE].segment_ids, [201, 211, 221, 232, 242])
    assert np.array_equal(gff_dict[RelativeLane.RIGHT_LANE].segment_ids, [201, 211, 220, 230, 240])


def test_getGeneralizedFrenetFrames_frenetStartsBehindAndEndsAheadOfCurrentLane_accurateFrameStartAndLength(
        scene_static_pg_split, route_plan_20_30):
    """
    test method _get_generalized_frenet_frames:
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
    gff = BehavioralGridState._get_generalized_frenet_frames(lane_id, station, route_plan_20_30)[RelativeLane.SAME_LANE]

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
def test_getGeneralizedFrenetFrames_AugmentedPartialCreatedWhenSplitEnds(left_right_lane_split_scene_static, route_plan_1_2_3):
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

    gff_dict = BehavioralGridState._get_generalized_frenet_frames(11, 900, route_plan_1_2_3, can_augment=can_augment)

    assert gff_dict[RelativeLane.LEFT_LANE].gff_type == GFFType.AugmentedPartial
    assert gff_dict[RelativeLane.RIGHT_LANE].gff_type == GFFType.AugmentedPartial
    assert gff_dict[RelativeLane.SAME_LANE].gff_type == GFFType.Partial
    assert gff_dict[RelativeLane.LEFT_LANE].has_segment_id(22)
    assert gff_dict[RelativeLane.RIGHT_LANE].has_segment_id(20)
    assert gff_dict[RelativeLane.SAME_LANE].has_segment_id(21)

def test_getDownstreamLaneSubsegments_planFiveOutOfTenSegments_validateTotalLengthAndOrdinal(scene_static_pg_split, route_plan_20_30):
    """
    test the method _get_downstream_lane_subsegments
        validate that total length of output sub segments == lookahead_dist;
    """

    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    road_ids = MapUtils.get_road_segment_ids()
    current_road_idx = 3
    current_ordinal = 1
    starting_lon = 20.
    lookahead_dist = 500.
    starting_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[current_road_idx])[current_ordinal]
    sub_segments, is_partial, is_augmented, _ = BehavioralGridState._get_downstream_lane_subsegments(starting_lane_id, starting_lon, lookahead_dist, route_plan_20_30)[RelativeLane.SAME_LANE]
    assert len(sub_segments) == 5
    for seg in sub_segments:
        assert MapUtils.get_lane_ordinal(seg.e_i_SegmentID) == current_ordinal
    tot_length = sum([seg.e_i_SEnd - seg.e_i_SStart for seg in sub_segments])
    assert np.isclose(tot_length, lookahead_dist)
    assert is_partial == False
    assert is_augmented == False


def test_getDownstreamLaneSubsegments_navPlanDoesNotFitMap_partialGeneralized(scene_static_pg_split, route_plan_20_30):
    """
    test the method _get_downstream_lane_subsegments
        add additional segment to nav_plan that does not exist on the map; validate a partial lookahead is done
    """

    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    road_segment_ids = MapUtils.get_road_segment_ids()
    current_road_idx = 3
    current_ordinal = 1
    starting_lon = 20.
    lookahead_dist = 600.
    starting_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_ids[current_road_idx])[current_ordinal]
    wrong_road_segment_id = 1234
    nav_plan_length = 8

    # Modify route plan for this test case
    route_plan = route_plan_20_30
    route_plan.s_Data.e_Cnt_num_road_segments = nav_plan_length + 1
    route_plan.s_Data.a_i_road_segment_ids = np.array(route_plan.s_Data.a_i_road_segment_ids[:nav_plan_length].tolist() +
                                                      [wrong_road_segment_id])
    route_plan.s_Data.a_Cnt_num_lane_segments = route_plan.s_Data.a_Cnt_num_lane_segments[:(nav_plan_length + 1)]
    route_plan.s_Data.as_route_plan_lane_segments = route_plan.s_Data.as_route_plan_lane_segments[:(nav_plan_length + 1)]

    lane_number = 1

    for lane_segment in route_plan.s_Data.as_route_plan_lane_segments[-1]:
        lane_segment.e_i_lane_segment_id = wrong_road_segment_id + lane_number
        lane_number += 1

    # test navigation plan fitting the lookahead distance, and add non-existing road at the end of the plan
    # validate getting the relevant exception
    subsegs, is_partial, _, _ = BehavioralGridState._get_downstream_lane_subsegments(starting_lane_id, starting_lon, lookahead_dist, route_plan)[RelativeLane.SAME_LANE]

    subseg_ids = [subseg.e_i_SegmentID for subseg in subsegs]

    # verify the wrong road segment is not added
    # make sure that the non-existent road segment is not contained in the GFF
    assert np.all([MapUtils.get_road_segment_id_from_lane_id(subseg.e_i_SegmentID) != wrong_road_segment_id for subseg in subsegs])
    # make sure that the previous existing road segments were used
    assert len(subsegs) == nav_plan_length - current_road_idx
    # make sure the lanes are in the correct order
    assert np.array_equal(subseg_ids, [231, 241, 251, 261, 271])
    # make sure the GFF created was of type Partial since it should not extend the entire route plan
    assert is_partial == True


def test_getDownstreamLaneSubsegments_navPlanTooShort_validateRelevantException(scene_static_pg_split, route_plan_20_30):
    """
    test the method _get_downstream_lane_subsegments
        test exception for too short nav plan; validate the relevant exception
    """

    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    road_segment_ids = MapUtils.get_road_segment_ids()
    current_road_idx = 3
    current_ordinal = 1
    starting_lon = 20.
    lookahead_dist = 500.
    starting_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_ids[current_road_idx])[current_ordinal]
    nav_plan_length = 7

    # Modify route plan for this test case
    route_plan = route_plan_20_30
    route_plan.s_Data.e_Cnt_num_road_segments = nav_plan_length
    route_plan.s_Data.a_i_road_segment_ids = route_plan.s_Data.a_i_road_segment_ids[:nav_plan_length]
    route_plan.s_Data.a_Cnt_num_lane_segments = route_plan.s_Data.a_Cnt_num_lane_segments[:nav_plan_length]
    route_plan.s_Data.as_route_plan_lane_segments = route_plan.s_Data.as_route_plan_lane_segments[:nav_plan_length]

    # test the case when the navigation plan is too short; validate the relevant exception
    try:
        BehavioralGridState._get_downstream_lane_subsegments(starting_lane_id, starting_lon, lookahead_dist, route_plan)
        assert False
    except NavigationPlanTooShort:
        assert True


def test_getDownstreamLaneSubsegments_lookAheadDistLongerThanMap_validatePartialGeneralized(scene_static_pg_split, route_plan_20_30):
    """
    test the method _get_downstream_lane_subsegments
        test exception for too short map but nav_plan is long enough; validate the relevant exception
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    road_segment_ids = MapUtils.get_road_segment_ids()
    current_road_idx = 9
    current_ordinal = 1
    starting_lon = 50.
    starting_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_ids[current_road_idx])[current_ordinal]
    lookadhead_dist = 1000

    # Modify route plan to make it extend past the last lane in the scene_static fixture
    route_plan = route_plan_20_30
    route_plan.s_Data.e_Cnt_num_road_segments += 1
    route_plan.s_Data.a_i_road_segment_ids = np.append(route_plan.s_Data.a_i_road_segment_ids, 30)
    route_plan.s_Data.a_Cnt_num_lane_segments = np.append(route_plan.s_Data.a_Cnt_num_lane_segments,
                                                          route_plan.s_Data.a_Cnt_num_lane_segments[-1])
    route_plan.s_Data.as_route_plan_lane_segments.append([RoutePlanLaneSegment(300,0,0),
                                                          RoutePlanLaneSegment(301,0,0),
                                                          RoutePlanLaneSegment(302,0,0)])

    # test the case when the map is too short; validate partial lookahead is done
    subsegs, is_partial, is_augmented, _ = BehavioralGridState._get_downstream_lane_subsegments(starting_lane_id, starting_lon, lookadhead_dist, route_plan)[RelativeLane.SAME_LANE]
    subseg_ids = [subseg.e_i_SegmentID for subseg in subsegs]

    # make sure the subsegments are in the correct order
    assert np.array_equal(subseg_ids, [291])
    # make sure the the gff is marked as partial
    assert is_partial == True
    assert is_augmented == False

def test_getUpstreamLaneSubsegments_backwardHorizonOnLane_NoUpstreamLaneSubsegments(scene_static_pg_split: SceneStatic):
    """
    Test _get_upstream_lane_subsegments
    The distance to travel backwards is small enough that it is still on the same lane. This should result in no upstream lane subsegments
    returned.
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    upstream_lane_subsegments = BehavioralGridState._get_upstream_lane_subsegments(200, 100, 50)
    assert upstream_lane_subsegments == []


def test_getUpstreamLaneSubsegments_backwardHorizonPassesBeginningOfLane_CorrectUpstreamLaneSubsegments(scene_static_pg_split: SceneStatic):
    """
    Test _get_upstream_lane_subsegments
    This tests the scenario where the backwards horizon extends to upstream lanes and they exist. The result should be the "normal" output
    with subsegments extending back as far as the provided horizon with start and end stations assigned accordingly. The expected values
    were calculated as follows:

        station_on_220 = 5
        backward_distance = 150
        length_of_200 = 120.84134201631973  (from pickle file)
        length_of_210 = 119.64304560784024  (from pickle file)

        start_station_on_200 = (length_of_200 + length_of_210 + station_on_220) - backward_distance
        end_station_on_200 = length_of_200

        start_station_on_210 = 0.0  (beginning of lane segment)
        end_station_on_210 = length_of_210
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    upstream_lane_subsegments = BehavioralGridState._get_upstream_lane_subsegments(220, 5, 150)

    # Check size
    assert len(upstream_lane_subsegments) == 2

    # Check order
    assert upstream_lane_subsegments[0].e_i_SegmentID == 200
    assert upstream_lane_subsegments[1].e_i_SegmentID == 210

    # Check start and end stations
    assert upstream_lane_subsegments[0].e_i_SStart == 95.48438762415998
    assert upstream_lane_subsegments[0].e_i_SEnd == 120.84134201631973

    assert upstream_lane_subsegments[1].e_i_SStart == 0.0
    assert upstream_lane_subsegments[1].e_i_SEnd == 119.64304560784024


def test_getUpstreamLaneSubsegments_NoUpstreamLane_validateUpstreamLaneNotFound(scene_static_pg_split: SceneStatic):
    """
    Test _get_upstream_lane_subsegments
    This tests the scenario where the backwards horizon extends to upstream lanes but an upstream lane doesn't exist at some point. With
    starting close to the beginning of lane 210 and going backwards 150 m, the beginning of lane 200 is passed. Since lane 200 doesn't
    have any upstream lanes, the UpstreamLaneNotFound exception should be raised.
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)

    try:
        BehavioralGridState._get_upstream_lane_subsegments(210, 5, 150)
        assert False
    except UpstreamLaneNotFound:
        assert True

def test_isObjectInLane_carOnLaneLine(scene_static_short_testable: SceneStatic):
    """
    Tests the method is_object_in_lane. Places a car in lane 11, shifted 1.5 meters left from lane 11's nominal points.
    :param scene_static_short_testable:
    :return:
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_short_testable)

    # Create car in lane 11 which is offset 1.5 meters to the left
    dyn_obj = DynamicObject.create_from_map_state(obj_id=10, timestamp=5, map_state=MapState(np.array([1.0,1,0,1.5,0,0]), 11),
                                                  size=ObjectSize(4,1.5,1), confidence=1, off_map=False)

    assert BehavioralGridState.is_object_in_lane(dyn_obj, 11) == True
    assert BehavioralGridState.is_object_in_lane(dyn_obj, 12) == True
    assert BehavioralGridState.is_object_in_lane(dyn_obj, 10) == False

def test_isObjectInLane_carInSingleLane(scene_static_short_testable: SceneStatic):

    SceneStaticModel.get_instance().set_scene_static(scene_static_short_testable)

    # Create car in lane 11
    dyn_obj = DynamicObject.create_from_map_state(obj_id=10, timestamp=5, map_state=MapState(np.array([1.0,1,0,0,0,0]), 11),
                                                  size=ObjectSize(1,1,1), confidence=1, off_map=False)

    assert BehavioralGridState.is_object_in_lane(dyn_obj, 11) == True
    assert BehavioralGridState.is_object_in_lane(dyn_obj, 12) == False
    assert BehavioralGridState.is_object_in_lane(dyn_obj, 10) == False

def test_isObjectInLane_laneSplit_carInOverlap(scene_static_oval_with_splits: SceneStatic):
    """
    Validate that projected object is correctly placed in overlapping lane
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_oval_with_splits)

    # Create other car in lane 21, which overlaps with lane 22
    dyn_obj = DynamicObject.create_from_map_state(obj_id=10, timestamp=5,
                                                  map_state=MapState(np.array([1,1,0,0,0,0]), 19670532),
                                                  size=ObjectSize(5, 2, 2), confidence=1, off_map=False)
    assert BehavioralGridState.is_object_in_lane(dyn_obj, 19670532) == True
    assert BehavioralGridState.is_object_in_lane(dyn_obj, 19670533) == True

def test_isObjectInLane_laneMerge_carInOverlap(scene_static_oval_with_splits: SceneStatic):
    """
    Validate that projected object is correctly placed in overlapping lane
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_oval_with_splits)

    # Create other car in lane 21, which overlaps with lane 22
    dyn_obj = DynamicObject.create_from_map_state(obj_id=10, timestamp=5,
                                                  map_state=MapState(np.array([1,1,0,0,0,0]), 58375684),
                                                  size=ObjectSize(5, 2, 2), confidence=1, off_map=False)
    assert BehavioralGridState.is_object_in_lane(dyn_obj, 58375684) == True
    assert BehavioralGridState.is_object_in_lane(dyn_obj, 58375685) == True

