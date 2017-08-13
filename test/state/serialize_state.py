from decision_making.src.state.enriched_state import *


def test_state_serialization():
    # Publish dummy state
    occupancy_state = EnrichedOccupancyState(np.array([0.0]), np.array([0.0]))
    static_objects = [
        EnrichedObjectState(0, 0, 0, 0, 0, 0, EnrichedObjectSize(0, 0, 0), EnrichedRoadLocalization(0, 0, 0, 0, 0),
                            0, 0)]
    dynamic_objects = [EnrichedDynamicObject(0, 0, 0, 0, 0, 0, EnrichedObjectSize(0, 0, 0),
                                             EnrichedRoadLocalization(0, 0, 0, 0, 0), 0, 0, 0, 0, 0, 0)]
    ego_state = EnrichedEgoState(0, 0, 0, 0, 0, 0, EnrichedObjectSize(0, 0, 0),
                                 EnrichedRoadLocalization(0, 0, 0, 0, 0), 0, 0, 0, 0, 0, 0, 0)
    perceived_road = EnrichedPerceivedRoad(0, [EnrichedLanesStructure(np.array([0.0]), np.array([0.0]))], 0)

    enriched_state = EnrichedState(occupancy_state=occupancy_state, static_objects=static_objects,
                                   dynamic_objects=dynamic_objects, ego_state=ego_state,
                                   perceived_road=perceived_road)

    enriched_state_serialized = enriched_state.serialize()
    print(enriched_state_serialized)


test_state_serialization()
