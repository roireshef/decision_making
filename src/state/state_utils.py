import numpy as np

from decision_making.src.planning.types import FP_SX, CartesianExtendedState
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.state.state import DynamicObject


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