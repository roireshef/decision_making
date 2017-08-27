from decision_making.src.planning.behavioral.policy import PolicyConfig


class DefaultPolicyConfig(PolicyConfig):
    def __init__(self, margin_from_road_edge: float = 0.2,
                 prefer_other_lanes_where_blocking_object_distance_less_than: float = 40.0,
                 prefer_other_lanes_if_improvement_is_greater_than: float = 5.0,
                 prefer_any_lane_center_if_blocking_object_distance_greater_than: float = 25.0,
                 assume_blocking_object_at_rear_if_distance_less_than: float = 0.0):
        self.margin_from_road_edge = margin_from_road_edge
        super().__init__()

        self.prefer_other_lanes_where_blocking_object_distance_less_than = \
            prefer_other_lanes_where_blocking_object_distance_less_than
        self.prefer_other_lanes_if_improvement_is_greater_than = \
            prefer_other_lanes_if_improvement_is_greater_than
        self.prefer_any_lane_center_if_blocking_object_distance_greater_than = \
            prefer_any_lane_center_if_blocking_object_distance_greater_than
        self.assume_blocking_object_at_rear_if_distance_less_than = \
            assume_blocking_object_at_rear_if_distance_less_than
