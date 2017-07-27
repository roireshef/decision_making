
from src.state.enriched_state import *


def test_init():
    size = EnrichedObjectSize(0, 0, 0)
    assert isinstance(size, EnrichedObjectSize)
    occ = EnrichedOccupancyState(np.array([]), np.array([]))
    assert isinstance(occ, EnrichedOccupancyState)
    loc = EnrichedRoadLocalization(0, 0, 0, 0, 0)
    assert isinstance(loc, EnrichedRoadLocalization)
    stat = EnrichedObjectState(0, 0, 0, 0, 0, 0, size, loc, 0, 0)
    assert isinstance(stat, EnrichedObjectState)
    dyn = EnrichedDynamicObject(0, 0, 0, 0, 0, 0, size, loc, 0, 0, 0, 0, 0, 0)
    assert isinstance(dyn, EnrichedDynamicObject)
    ego = EnrichedEgoState(0, 0, 0, 0, 0, 0, size, loc, 0, 0, 0, 0, 0, 0, 0)
    assert isinstance(ego, EnrichedEgoState)
    lane = EnrichedLanesStructure(np.array([]), np.array([]))
    assert isinstance(lane, EnrichedLanesStructure)
    road = EnrichedPerceivedRoad(0, [lane], 0)
    assert isinstance(road, EnrichedPerceivedRoad)
    state = EnrichedState(occ, [stat], [dyn], ego, road)
    assert isinstance(state, EnrichedState)
