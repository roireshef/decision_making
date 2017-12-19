from decision_making.src.state.state import ObjectSize, OccupancyState, DynamicObject, EgoState, State
from mapping.src.transformations.geometry_utils import CartesianFrame
import numpy as np

def test_init():

    rel_pos = np.array([1, 1, 0])
    rel_yaw = 0
    ego_pos = np.array([1, 0, 0])
    ego_yaw = -np.pi/2
    glob_pos, glob_yaw = CartesianFrame.convert_relative_to_global_frame(rel_pos, rel_yaw, ego_pos, ego_yaw)

    # test state.predict
    size = ObjectSize(0, 0, 0)
    occ = OccupancyState(0, np.array([[1,1,0]]), np.array([0]))
    dyn = DynamicObject(1, 0, 15, 1, 0, 0.1, size, 0, 10, 1, 0, 0)
    ego = EgoState(0, 0, 5, 0, 0, 0, size, 0, 0, 2, 0, 0, 0)
    state = State(occ, [dyn], ego)
    state.ego_state.acceleration_lon = 1
    state.ego_state.turn_radius = 1000
    state.dynamic_objects[0].acceleration_lon = -1

