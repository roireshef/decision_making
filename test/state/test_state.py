
from decision_making.src.state.state import *


def test_init():
    size = ObjectSize(0, 0, 0)
    assert isinstance(size, ObjectSize)
    occ = OccupancyState(0, np.array([]), np.array([]))
    assert isinstance(occ, OccupancyState)
    loc = RoadLocalization(0, 0, 0, 0, 0, 0, 0)
    assert isinstance(loc, RoadLocalization)
    dyn = DynamicObject(0, 0, 0, 0, 0, 0, size, loc, 0, 0, 0, 0, 0, 0, 0)
    assert isinstance(dyn, DynamicObject)
    ego = EgoState(0, 0, 0, 0, 0, 0, size, loc, 0, 0, 0, 0, 0, 0, 0, 0)
    assert isinstance(ego, EgoState)
    lane = LanesStructure(np.array([]), np.array([]))
    assert isinstance(lane, LanesStructure)
    road = PerceivedRoad(0, [lane], 0)
    assert isinstance(road, PerceivedRoad)
    state = State(occ, [dyn], ego, road)
    assert isinstance(state, State)

test_init()
