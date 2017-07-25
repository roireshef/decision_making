
from state.enriched_state import *

size = EnrichedObjectSize(0, 0, 0)
occ = EnrichedOccupancyState(np.array([]), np.array([]))
loc = EnrichedRoadLocalization(0, 0, 0, 0, 0)
stat = EnrichedObjectState(0, 0, 0, 0, 0, 0, size, loc, 0, 0)
dyn = EnrichedDynamicObject(0, 0, 0, 0, 0, 0, size, loc, 0, 0, 0, 0, 0, 0)
ego = EnrichedEgoState(0, 0, 0, 0, 0, 0, size, loc, 0, 0, 0, 0, 0, 0, 0)
lane = EnrichedLanesStructure(np.array([]), np.array([]))
road = EnrichedPerceivedRoad(0, [lane], 0)
state = EnrichedState(occ, [stat], [dyn], ego, road)
