from common_data.interface.Rte_Types.python.sub_structures.TsSYS_MapState import TsSYSMapState

from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from decision_making.src.planning.types import FrenetState2D, FS_SX, FS_DX
from decision_making.src.utils.map_utils import MapUtils
from mapping.src.model.constants import ROAD_SHOULDERS_WIDTH


class MapState(PUBSUB_MSG_IMPL):
    a_LaneFState = FrenetState2D
    e_i_LaneID = int

    def __init__(self, lane_fstate: FrenetState2D, lane_id: int) -> None:
        self.a_LaneFState = lane_fstate
        self.e_i_LaneID = lane_id

    def is_on_road(self) -> bool:
        """
        Returns true of the object is on the road. False otherwise.
        :return: Returns true of the object is on the road. False otherwise.
        """
        on_road_longitudinally = (0 <= self.a_LaneFState[FS_SX] < MapUtils.get_lane_length(self.e_i_LaneID))
        dist_from_right, dist_from_left = MapUtils.get_dist_to_lane_borders(self.e_i_LaneID, self.a_LaneFState[FS_SX])
        on_road_laterally = (-dist_from_right - ROAD_SHOULDERS_WIDTH < self.a_LaneFState[FS_DX] < dist_from_left + ROAD_SHOULDERS_WIDTH)
        return on_road_longitudinally and on_road_laterally

    def serialize(self)-> TsSYSMapState:
        pubsub_msg = TsSYSMapState()
        pubsub_msg.a_LaneFState = self.a_LaneFState
        pubsub_msg.e_i_LaneID = self.e_i_LaneID
        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsub_msg: TsSYSMapState)-> ():
        return cls(pubsub_msg.a_LaneFState, pubsub_msg.e_i_LaneID)
