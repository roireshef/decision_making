import numpy as np
from gm_lcm import LcmMapState

from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from decision_making.src.planning.types import FrenetState2D
from decision_making.src.utils.lcm_utils import LCMUtils


class MapState(PUBSUB_MSG_IMPL):
    road_fstate = FrenetState2D
    road_id = int

    def __init__(self, road_fstate, road_id):
        # type: (FrenetState2D, int) -> MapState
        self.road_fstate = road_fstate
        self.road_id = road_id

    #TODO: implement. Consider whether this is a property of map state or a different function in Map Utils.
    @property
    def lane_center_lat(self):
        pass

    # TODO: implement
    @property
    def intra_lane_lat(self) -> int:
        pass

    #TODO: implement lane number computation from map and fstate
    @property
    def lane_num(self) -> int:
        pass

    def serialize(self):
        # type: () -> LcmMapState
        lcm_msg = LcmMapState()
        lcm_msg.road_fstate = LCMUtils.numpy_array_to_lcm_non_typed_numpy_array(self.road_fstate)
        lcm_msg.road_id = self.road_id
        return lcm_msg

    @classmethod
    def deserialize(cls, lcm_msg):
        # type: (LcmMapState) -> MapState
        return cls(np.ndarray(shape=tuple(lcm_msg.lane_state.shape)
                              , buffer=np.array(lcm_msg.lane_state.data)
                              , dtype=float)
        ,lcm_msg.road_id)

