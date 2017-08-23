from decision_making.src.map.map_model import MapModel
from decision_making.src.state.state import *


def test_init():
    # test state.predict
    map_model = MapModel()
    map_api = MapAPI(map_model)

    size = ObjectSize(0, 0, 0)
    occ = OccupancyState(0, np.array([[1,1,0]]), np.array([0]))
    dyn = DynamicObject(1, 0, 15, 1, 0, 0.1, size, 0, 10, 1, 0, 0, map_api)
    ego = EgoState(0, 0, 5, 0, 0, 0, size, 0, 0, 2, 0, 0, 0, map_api)
    state = State(occ, [dyn], ego)
    state.ego_state.acceleration_lon = 1
    state.ego_state.turn_radius = 1000
    state.dynamic_objects[0].acceleration_lon = -1
    state.predict(2000, map_api)

test_init()
