import numpy as np
from common_data.interface.py.idl_generated_files.dm.sub_structures.LcmMapState import LcmMapState

from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from decision_making.src.planning.types import FrenetState2D, FS_DX
from mapping.src.service.map_service import MapService


class MapState(PUBSUB_MSG_IMPL):
    road_fstate = FrenetState2D
    road_id = int

    def __init__(self, road_fstate, road_id):
        # type: (FrenetState2D, int) -> MapState
        self.road_fstate = road_fstate
        self.road_id = road_id

    @property
    def lane_center_lat(self):
        lane_width, _, lane_num = self._get_current_lane_params()
        return (lane_num+0.5)*lane_width

    @property
    def intra_lane_lat(self) -> int:
        lane_width, lat_pos_from_right, lane_num = self._get_current_lane_params()
        return lat_pos_from_right - lane_num * lane_width

    @property
    def lane_num(self) -> int:
        _, _, lane = self._get_current_lane_params()
        return lane

    def _get_current_lane_params(self):
        """
        :return: A tuple consisting of: (the lane width,lateral position in frenet from right hand side of road,lane number between 0 and num_lanes-1)
        """
        # type: MapState -> (float, float, int)
        lane_width = MapService.get_instance().get_road(self.road_id).lane_width
        lat_pos_from_right = self.road_fstate[FS_DX]
        lane = int(np.math.floor(lat_pos_from_right / lane_width))
        return lane_width, lat_pos_from_right, lane

    def serialize(self):
        # type: () -> LcmMapState
        lcm_msg = LcmMapState()
        lcm_msg.road_fstate = self.road_fstate
        lcm_msg.road_id = self.road_id
        return lcm_msg

    @classmethod
    def deserialize(cls, lcm_msg):
        # type: (LcmMapState) -> MapState
        return cls(lcm_msg.road_fstate.data, lcm_msg.road_id)
