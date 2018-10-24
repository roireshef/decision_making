import numpy as np
from common_data.lcm.generatedFiles.gm_lcm import LcmMapState

from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from decision_making.src.planning.types import FrenetState2D
from common_data.lcm.python.utils.lcm_utils import LCMUtils
from mapping.src.service.map_service import MapService


class MapState(PUBSUB_MSG_IMPL):
    lane_fstate = FrenetState2D
    lane_id = int

    def __init__(self, lane_fstate, lane_id):
        # type: (FrenetState2D, int) -> MapState
        self.lane_fstate = lane_fstate
        self.lane_id = lane_id

    @property
    def lane_num(self):
        # type: (int) -> int
        return MapService().get_instance().get_lane_index(self.lane_id)

    def serialize(self):
        # type: () -> LcmMapState
        lcm_msg = LcmMapState()
        lcm_msg.road_fstate = LCMUtils.numpy_array_to_lcm_non_typed_numpy_array(self.lane_fstate)
        lcm_msg.road_id = self.lane_id
        return lcm_msg

    @classmethod
    def deserialize(cls, lcm_msg):
        # type: (LcmMapState) -> MapState
        return cls(np.ndarray(shape=tuple(lcm_msg.road_fstate.shape)
                              , buffer=np.array(lcm_msg.road_fstate.data), dtype=float), lcm_msg.road_id)
