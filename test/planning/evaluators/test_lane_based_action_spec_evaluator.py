from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import RelativeLane
from decision_making.src.planning.behavioral.evaluators.augmented_lane_action_spec_evaluator import AugmentedLaneActionSpecEvaluator

from unittest.mock import patch

from decision_making.test.planning.behavioral.behavioral_state_fixtures import state_with_lane_split_on_right, state_with_lane_split_on_left, \
    state_with_lane_split_on_left_and_right
from decision_making.test.planning.custom_fixtures import route_plan_lane_split_on_right, route_plan_lane_split_on_left, route_plan_1_2, \
    route_plan_lane_split_on_left_and_right


def test_findMinCostAugmentedLane_rightSplitLowerCost(state_with_lane_split_on_right, route_plan_lane_split_on_right):
    behavioral_grid_state = BehavioralGridState.create_from_state(state_with_lane_split_on_right, route_plan_lane_split_on_right, None)
    evaluator = AugmentedLaneActionSpecEvaluator(None)
    # set route cost of lane 21 to have a high cost, leading to the right split maneuver towards lane 20 to be cheaper
    route_plan_lane_split_on_right.s_Data.as_route_plan_lane_segments[1][1].e_cst_lane_end_cost = 1

    min_cost_lane = evaluator._find_min_cost_augmented_lane(behavioral_grid_state, route_plan_lane_split_on_right)
    assert min_cost_lane == RelativeLane.RIGHT_LANE


def test_findMinCostAugmentedLane_rightSplitHigherCost(state_with_lane_split_on_right, route_plan_lane_split_on_right):
    behavioral_grid_state = BehavioralGridState.create_from_state(state_with_lane_split_on_right, route_plan_lane_split_on_right, None)
    evaluator = AugmentedLaneActionSpecEvaluator(None)
    # set route cost of lane 20 to have a high cost, leading to the same lane maneuver towards lane 21 to be cheaper
    route_plan_lane_split_on_right.s_Data.as_route_plan_lane_segments[1][0].e_cst_lane_end_cost = 1

    min_cost_lane = evaluator._find_min_cost_augmented_lane(behavioral_grid_state, route_plan_lane_split_on_right)
    assert min_cost_lane == RelativeLane.SAME_LANE


def test_findMinCostAugmentedLane_leftSplitLowerCost(state_with_lane_split_on_left, route_plan_lane_split_on_left):
    behavioral_grid_state = BehavioralGridState.create_from_state(state_with_lane_split_on_left, route_plan_lane_split_on_left, None)
    evaluator = AugmentedLaneActionSpecEvaluator(None)
    # set route cost of lane 21 to have a high cost, leading to the left split lane maneuver towards lane 22 to be cheaper
    route_plan_lane_split_on_left.s_Data.as_route_plan_lane_segments[1][1].e_cst_lane_end_cost = 1

    min_cost_lane = evaluator._find_min_cost_augmented_lane(behavioral_grid_state, route_plan_lane_split_on_left)
    assert min_cost_lane == RelativeLane.LEFT_LANE

def test_findMinCostAugmentedLane_leftSplitHigherCost(state_with_lane_split_on_left, route_plan_lane_split_on_left):
    behavioral_grid_state = BehavioralGridState.create_from_state(state_with_lane_split_on_left,
                                                                  route_plan_lane_split_on_left, None)
    evaluator = AugmentedLaneActionSpecEvaluator(None)
    # set route cost of lane 22 to have a high cost, leading to the same lane lane maneuver towards lane 21 to be cheaper
    route_plan_lane_split_on_left.s_Data.as_route_plan_lane_segments[1][2].e_cst_lane_end_cost = 1

    min_cost_lane = evaluator._find_min_cost_augmented_lane(behavioral_grid_state, route_plan_lane_split_on_left)
    assert min_cost_lane == RelativeLane.SAME_LANE

@patch('decision_making.src.planning.behavioral.evaluators.lane_based_action_spec_evaluator.PREFER_LEFT_SPLIT_OVER_RIGHT_SPLIT', False)
def test_findMinCostAugmentedLane_leftRightSplit_usePreference(state_with_lane_split_on_left_and_right, route_plan_lane_split_on_left_and_right):
    behavioral_grid_state = BehavioralGridState.create_from_state(state_with_lane_split_on_left_and_right,
                                                                  route_plan_lane_split_on_left_and_right, None)
    evaluator = AugmentedLaneActionSpecEvaluator(None)

    # based on PREFER_LEFT_SPLIT_OVER_RIGHT_SPLIT = False, the right split should be taken when the straight lane 21 has a high cost
    route_plan_lane_split_on_left_and_right.s_Data.as_route_plan_lane_segments[1][1].e_cst_lane_end_cost = 1

    min_cost_lane = evaluator._find_min_cost_augmented_lane(behavioral_grid_state, route_plan_lane_split_on_left_and_right)
    assert min_cost_lane == RelativeLane.RIGHT_LANE


