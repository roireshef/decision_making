import numpy as np
from decision_making.src.planning.types import FP_SX, FrenetState2D, FS_SX
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.state.state import DynamicObject, EgoState
from mapping.src.service.map_service import MapService


class MapUtils:
    # TODO: replace with navigation plan aware function from map API
    @staticmethod
    def get_road_rhs_frenet(obj: DynamicObject):
        return MapService.get_instance()._rhs_roads_frenet[obj.road_localization.road_id]

    # TODO: replace this call with the road localization once it is updated to be hold a frenet state
    @staticmethod
    def get_object_road_localization(obj: DynamicObject, road_frenet: FrenetSerret2DFrame) -> FrenetState2D:
        target_obj_fpoint = road_frenet.cpoint_to_fpoint(np.array([obj.x, obj.y]))
        obj_init_fstate = road_frenet.cstate_to_fstate(np.array([
            obj.x, obj.y,
            obj.yaw,
            obj.total_speed,
            obj.acceleration_lon,
            road_frenet.get_curvature(target_obj_fpoint[FP_SX])  # We zero the object's curvature relative to the road
        ]))
        return obj_init_fstate

    @staticmethod
    def get_ego_road_localization(ego: EgoState, road_frenet: FrenetSerret2DFrame):
        ego_init_cstate = np.array([ego.x, ego.y, ego.yaw, ego.v_x, ego.acceleration_lon, ego.curvature])
        return road_frenet.cstate_to_fstate(ego_init_cstate)