from decision_making.src.state.state import *


def test_state_serialization():
    # Publish dummy state
    occupancy_state = OccupancyState(np.array([0.0]), np.array([0.0]))
    dynamic_objects = [DynamicObject(0, 0, 0, 0, 0, 0, ObjectSize(0, 0, 0),
                                             RoadLocalization(0, 0, 0, 0, 0), 0, 0, 0, 0, 0, 0)]
    ego_state = EgoState(0, 0, 0, 0, 0, 0, ObjectSize(0, 0, 0),
                                 RoadLocalization(0, 0, 0, 0, 0), 0, 0, 0, 0, 0, 0, 0)
    perceived_road = PerceivedRoad(0, [LanesStructure(np.array([0.0]), np.array([0.0]))], 0)

    state = State(occupancy_state=occupancy_state,
                                   dynamic_objects=dynamic_objects, ego_state=ego_state,
                                   perceived_road=perceived_road)

    state_serialized = state.serialize()
    print(state_serialized)


test_state_serialization()
