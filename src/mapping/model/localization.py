from decision_making.src.global_constants import PUBSUB_MSG_IMPL


class RoadLocalization(PUBSUB_MSG_IMPL):
    def __init__(self, road_id, lane_num, intra_road_lat, intra_lane_lat, road_lon, intra_road_yaw):
        # type: (int, int, float, float, float, float, float, float) -> None
        """
        object that holds localization of an entity relative to some road on the map
        :param road_id:
        :param lane_num: 0 is the rightmost
        :param intra_road_lat: in meters; latitude from the right edge of the road
        :param intra_lane_lat: in meters, latitude from the right edge of the lane
        :param road_lon: in meters, longitude relatively to the road start
        :param intra_road_yaw: 0 is along road's local tangent
        """
        self.road_id = road_id
        self.lane_num = lane_num
        self.intra_road_lat = intra_road_lat
        self.intra_lane_lat = intra_lane_lat
        self.road_lon = road_lon
        self.intra_road_yaw = intra_road_yaw


class RoadCoordinatesDifference:
    def __init__(self, rel_lat, rel_lon, rel_yaw, rel_lane):
        # type: (float, float, float, int) -> None
        """
        this object holds the difference between (two) road-coordinates (latitude, longitude and yaw).
        (this is not relative to a specific road)
        :param rel_lane: in lane, lane relatively to ego
        :param rel_lat: in meters, latitude relatively to ego
        :param rel_lon: in meters, longitude relatively to ego
        :param rel_yaw: in radians, yaw relatively to ego
        """
        self.rel_lane = rel_lane
        self.rel_lat = rel_lat
        self.rel_lon = rel_lon
        self.rel_yaw = rel_yaw
