from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg

class TempRoutePlanner:
    """
    This class holds a temporary implementation of RoutePlanner to unblock development of lane splits feature.
    """
    @staticmethod
    def get_cost(lane_id: int, navigation_plan: NavigationPlanMsg) -> int:
        """
        Returns cost for traversing input lane_id
        :param lane_id: ID of lane whose cost we want
        :param navigation_plan: NavigationPlanMsg, temporary param for navigation plan 
        :return: Cost of lane traversal for input lane_id
        """
        # Import here to avoid circular import dependency
        from decision_making.src.utils.map_utils import MapUtils

        for road_seg_id in navigation_plan.road_ids:
            for nav_lane_id in MapUtils.get_lanes_ids_from_road_segment_id(road_seg_id):
                if lane_id == nav_lane_id: return 0
        return 1