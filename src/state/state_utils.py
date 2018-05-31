import numpy as np

from decision_making.src.planning.types import FP_SX, CartesianExtendedState
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.state.state import DynamicObject


def get_object_cstate(object_state: DynamicObject, frenet_frame: FrenetSerret2DFrame) -> CartesianExtendedState:
    """
    Get objects extended Cartesian state
    :param object_state: object state
    :param frenet_frame: the Frenet frame of the road/lane relative to which we will calculate object's road parameters
        (as curvature)
    :return: objects extended Cartesian state
    """
    target_obj_fpoint = frenet_frame.cpoint_to_fpoint(np.array([object_state.x, object_state.y]))
    _, _, _, road_curvature_at_obj_location, _ = frenet_frame._taylor_interp(target_obj_fpoint[FP_SX])
    velocity_yaw = np.arctan2(object_state.v_y, object_state.v_x)
    object_acceleration = 0.0

    # Object's Frenet state
    obj_cstate = np.array([
        object_state.x,
        object_state.y,
        velocity_yaw,
        object_state.total_speed,
        object_acceleration,
        road_curvature_at_obj_location  # We don't care about other agent's curvature, only the road's
    ])

    return obj_cstate


def get_object_fstate(object_state: DynamicObject, frenet_frame: FrenetSerret2DFrame) -> CartesianExtendedState:
    """
    Get object's Frenet state
    :param object_state: object state
    :param frenet_frame: the Frenet frame of the road/lane relative to which we will calculate object's state parameters
    :return: objects Frenet state
    """
    obj_cstate = get_object_cstate(object_state=object_state, frenet_frame=frenet_frame)
    obj_fstate = frenet_frame.cstate_to_fstate(cstate=obj_cstate)

    return obj_fstate