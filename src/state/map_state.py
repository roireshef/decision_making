from interface.Rte_Types.python.sub_structures.TsSYS_MapState import TsSYSMapState

from decision_making.src.global_constants import PUBSUB_MSG_IMPL, ROAD_SHOULDERS_WIDTH
from decision_making.src.planning.types import FrenetState2D, FS_SX, FS_DX
from decision_making.src.utils.map_utils import MapUtils


class MapState(PUBSUB_MSG_IMPL):
    lane_fstate = FrenetState2D
    lane_id = int

    def __init__(self, lane_fstate: FrenetState2D, lane_id: int) -> None:
        self.lane_fstate = lane_fstate
        self.lane_id = lane_id

    def is_on_road(self) -> bool:
        """
        Returns true of the object is on the road. False otherwise.
        :return: Returns true of the object is on the road. False otherwise.
        """
        on_road_longitudinally = (0 <= self.lane_fstate[FS_SX] < MapUtils.get_lane_length(self.lane_id))
        dist_from_right, dist_from_left = MapUtils.get_dist_to_lane_borders(self.lane_id, self.lane_fstate[FS_SX])
        on_road_laterally = (-dist_from_right - ROAD_SHOULDERS_WIDTH < self.lane_fstate[FS_DX] < dist_from_left + ROAD_SHOULDERS_WIDTH)
        return on_road_longitudinally and on_road_laterally

    def serialize(self)-> TsSYSMapState:
        pubsub_msg = TsSYSMapState()
        pubsub_msg.a_LaneFState = self.lane_fstate
        pubsub_msg.e_i_LaneID = self.lane_id
        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsub_msg: TsSYSMapState)-> ():
        return cls(pubsub_msg.a_LaneFState, pubsub_msg.e_i_LaneID)
