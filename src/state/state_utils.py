from decision_making.src.planning.types import CartesianExtendedState
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.state.state import NewDynamicObject


def get_object_fstate(object_state: NewDynamicObject, frenet_frame: FrenetSerret2DFrame) -> CartesianExtendedState:
    """
    Get object's Frenet state
    :param object_state: object state
    :param frenet_frame: the Frenet frame of the road/lane relative to which we will calculate object's state parameters
    :return: objects Frenet state
    """

    return object_state.map_state.road_fstate
