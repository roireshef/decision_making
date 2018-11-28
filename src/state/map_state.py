import numpy as np
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.LcmMapState import LcmMapState

from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from decision_making.src.planning.types import FrenetState2D, CartesianExtendedState, C_X, C_Y, FS_SX, FS_DX
from common_data.lcm.python.utils.lcm_utils import LCMUtils
from decision_making.src.utils.map_utils import MapUtils
from mapping.src.model.constants import ROAD_SHOULDERS_WIDTH


class MapState(PUBSUB_MSG_IMPL):
    lane_fstate = FrenetState2D
    lane_id = int

    def __init__(self, lane_fstate, lane_id):
        # type: (FrenetState2D, int) -> MapState
        self.lane_fstate = lane_fstate
        self.lane_id = lane_id

    @classmethod
    def from_cartesian_state(cls, cartesian_state: CartesianExtendedState):
        # type: (CartesianExtendedState) -> MapState
        closest_lane_id = MapUtils.get_closest_lane(cartesian_state[:(C_Y+1)])
        lane_frenet = MapUtils.get_lane_frenet_frame(closest_lane_id)
        obj_fstate = lane_frenet.cstate_to_fstate(cartesian_state)
        return cls(obj_fstate, closest_lane_id)

    def to_cartesian_state(self):
        # type: () -> CartesianExtendedState
        lane_frenet = MapUtils.get_lane_frenet_frame(self.lane_id)
        return lane_frenet.fstate_to_cstate(self.lane_fstate)

    def is_on_road(self):
        # type: () -> bool
        """
        Returns true of the object is on the road. False otherwise.
        :return: Returns true of the object is on the road. False otherwise.
        """
        on_road_longitudinally = (0 <= self.lane_fstate[FS_SX] < MapUtils.get_lane_length(self.lane_id))
        dist_from_right, dist_from_left = MapUtils.get_dist_to_lane_borders(self.lane_id, self.lane_fstate[FS_SX])
        on_road_laterally = (-dist_from_right - ROAD_SHOULDERS_WIDTH < self.lane_fstate[FS_DX] < dist_from_left + ROAD_SHOULDERS_WIDTH)
        return on_road_longitudinally and on_road_laterally

    def serialize(self):
        # type: () -> LcmMapState
        lcm_msg = LcmMapState()
        lcm_msg.lane_fstate = self.lane_fstate
        lcm_msg.lane_id = self.lane_id
        return lcm_msg

    @classmethod
    def deserialize(cls, lcm_msg):
        # type: (LcmMapState) -> MapState
        return cls(lcm_msg.lane_fstate.data, lcm_msg.lane_id)
