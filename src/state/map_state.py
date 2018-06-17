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

    # TODO: implement. Consider whether this is a property of map state or a different function in Map Utils.
    @property
    def lane_center_lat(self):
        lane_width = MapService.get_instance().get_road(self.road_id).lane_width
        lat = self.road_fstate[FS_DX]
        lane = np.math.floor(lat / lane_width)
        return (lane+0.5)*lane_width

    # TODO: implement
    @property
    def intra_lane_lat(self) -> int:
        lane_width = MapService.get_instance().get_road(self.road_id).lane_width
        lat = self.road_fstate[FS_DX]
        lane = np.math.floor(lat / lane_width)
        return lat - lane * lane_width

    # TODO: implement lane number computation from map and fstate
    @property
    def lane_num(self) -> int:
        lane_width = MapService.get_instance().get_road(self.road_id).lane_width
        lat = self.road_fstate[FS_DX]
        return int(np.math.floor(lat / lane_width))

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
