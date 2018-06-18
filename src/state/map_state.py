import numpy as np
from common_data.lcm.generatedFiles.gm_lcm import LcmMapState

from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from decision_making.src.planning.types import FrenetState2D, FS_DX
from decision_making.src.utils.lcm_utils import LCMUtils
from mapping.src.service.map_service import MapService


class MapState(PUBSUB_MSG_IMPL):
    road_fstate = FrenetState2D
    road_id = int

    def __init__(self, road_fstate, road_id):
        # type: (FrenetState2D, int) -> MapState
        self.road_fstate = road_fstate
        self.road_id = road_id

    def get_current_lane_params(self):
        # type: (MapState) -> (float, float, int)
        lane_width = MapService.get_instance().get_road(self.road_id).lane_width
        lat = self.road_fstate[FS_DX]
        lane = int(np.math.floor(lat / lane_width))
        return lane_width, lat, lane

    @property
    def lane_center_lat(self):
        lane_width, _, lane = self.get_current_lane_params()
        return (lane+0.5)*lane_width

    @property
    def intra_lane_lat(self) -> int:
        lane_width, lat, lane = self.get_current_lane_params()
        return lat - lane * lane_width

    @property
    def lane_num(self) -> int:
        _, _, lane = self.get_current_lane_params()
        return lane

    def serialize(self):
        # type: () -> LcmMapState
        lcm_msg = LcmMapState()
        lcm_msg.road_fstate = LCMUtils.numpy_array_to_lcm_non_typed_numpy_array(self.road_fstate)
        lcm_msg.road_id = self.road_id
        return lcm_msg

    @classmethod
    def deserialize(cls, lcm_msg):
        # type: (LcmMapState) -> MapState
        return cls(np.ndarray(shape=tuple(lcm_msg.road_fstate.shape)
                              , buffer=np.array(lcm_msg.road_fstate.data), dtype=float), lcm_msg.road_id)
