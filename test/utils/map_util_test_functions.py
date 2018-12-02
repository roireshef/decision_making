from typing import List


from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.behavioral.data_objects import RelativeLane
from decision_making.src.planning.types import CartesianPoint2D
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.utils.map_utils import MapUtils

class TestFunctions:
    @staticmethod
    def test_get_road_rhs_frenet(road_id: int, expected_answer: FrenetSerret2DFrame) -> bool:
        test_answer = MapUtils.get_road_rhs_frenet(road_id)
        test_passed = (test_answer == expected_answer)
        if test_passed:
            print("Test get_road_segment_id_from_lane_id PASSED")
        else:
            print("Test get_road_segment_id_from_lane_id FAILED")
        return test_passed

    @staticmethod
    def test_get_lookahead_frenet_frame(lane_id: int, starting_lon: float, lookahead_dist: float,
                                        navigation_plan: NavigationPlanMsg, 
                                        expected_answer: GeneralizedFrenetSerretFrame) -> bool:
        test_answer = MapUtils.get_lookahead_frenet_frame(lane_id, starting_lon, lookahead_dist, navigation_plan)
        test_passed = (test_answer == expected_answer)
        if test_passed:
            print("Test get_lookahead_frenet_frame PASSED")
        else:
            print("Test get_lookahead_frenet_frame FAILED")
        return test_passed

    @staticmethod
    def test_get_road_segment_id_from_lane_id(lane_id: int, 
                                            expected_answer: int) -> bool:
        test_answer = MapUtils.get_road_segment_id_from_lane_id(lane_id)
        test_passed = (test_answer == expected_answer)
        if test_passed:
            print("Test get_road_segment_id_from_lane_id PASSED")
        else:
            print("Test get_road_segment_id_from_lane_id FAILED")
        return test_passed

    @staticmethod
    def test_get_lane_ordinal(lane_id: int, expected_answer: int) -> bool:
        test_answer = MapUtils.get_lane_ordinal(lane_id)
        test_passed = (test_answer == expected_answer)
        if test_passed:
            print("Test get_lane_ordinal PASSED")
        else:
            print("Test get_lane_ordinal FAILED")
        return test_passed

    @staticmethod
    def test_get_lane_length(lane_id: int, expected_answer: int) -> bool:
        test_answer = MapUtils.get_lane_length(lane_id)
        test_passed = (test_answer == expected_answer)
        if test_passed:
            print("Test get_lane_length PASSED")
        else:
            print("Test get_lane_length FAILED")
        return test_passed

    @staticmethod
    def test_get_lane_frenet_frame(lane_id: int, expected_answer: int) -> bool:
        test_answer = MapUtils.get_lane_frenet_frame(lane_id)
        test_passed = (test_answer == expected_answer)
        if test_passed:
            print("Test get_lane_frenet_frame PASSED")
        else:
            print("Test get_lane_frenet_frame FAILED")
        return test_passed

    @staticmethod
    def test_get_adjacent_lanes(lane_id: int, relative_lane: RelativeLane,
                                expected_answer: List[int]) -> bool:
        test_answer = MapUtils.get_adjacent_lanes(lane_id, relative_lane)
        test_passed = (test_answer == expected_answer)
        if test_passed:
            print("Test get_adjacent_lanes PASSED")
        else:
            print("Test get_adjacent_lanes FAILED")
        return test_passed

    @staticmethod
    def test_get_dist_from_lane_center_to_lane_borders(lane_id: int, s: float, expected_answer: (float, float)) -> bool:
        test_answer = MapUtils.get_dist_from_lane_center_to_lane_borders(lane_id, s)
        test_passed = (test_answer == expected_answer)
        if test_passed:
            print("Test get_dist_from_lane_center_to_lane_borders PASSED")
        else:
            print("Test get_dist_from_lane_center_to_lane_borders FAILED")
        return test_passed

    @staticmethod
    def test_get_dist_from_lane_center_to_road_borders(lane_id: int, s: float, expected_answer: (float, float)) -> bool:
        test_answer = MapUtils.get_dist_from_lane_center_to_road_borders(lane_id, s)
        test_passed = (test_answer == expected_answer)
        if test_passed:
            print("Test get_dist_from_lane_center_to_road_borders PASSED")
        else:
            print("Test get_dist_from_lane_center_to_road_borders FAILED")
        return test_passed

    @staticmethod
    def test_get_lane_width(lane_id: int, s: float, expected_answer: float) -> bool:
        test_answer = MapUtils.get_lane_width(lane_id, s)
        test_passed = (test_answer == expected_answer)
        if test_passed:
            print("Test get_lane_width PASSED")
        else:
            print("Test get_lane_width FAILED")
        return test_passed

    @staticmethod
    def test_get_upstream_lanes(lane_id: int, expected_answer: List[int]) -> bool:
        test_answer = MapUtils.get_upstream_lanes(lane_id)
        test_passed = (test_answer == expected_answer)
        if test_passed:
            print("Test get_upstream_lanes PASSED")
        else:
            print("Test get_upstream_lanes FAILED")
        return test_passed

    @staticmethod
    def test_get_downstream_lanes(lane_id: int, expected_answer: List[int])-> bool:
        test_answer = MapUtils.get_downstream_lanes(lane_id)
        test_passed = (test_answer == expected_answer)
        if test_passed:
            print("Test get_downstream_lanes PASSED")
        else:
            print("Test get_downstream_lanes FAILED")
        return test_passed

    @staticmethod
    def test_get_lanes_id_from_road_segment_id(road_segment_id: int, expected_answer: List[int])-> bool:
        test_answer = MapUtils.get_lanes_id_from_road_segment_id(road_segment_id)
        test_passed = (test_answer == expected_answer)
        if test_passed:
            print("Test get_lanes_id_from_road_segment_id PASSED")
        else:
            print("Test get_lanes_id_from_road_segment_id FAILED")
        return test_passed