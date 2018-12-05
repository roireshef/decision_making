from decision_making.src.planning.behavioral.data_objects import RelativeLane
from decision_making.src.planning.types import FS_SX

from decision_making.test.planning.behavioral.behavioral_state_fixtures import behavioral_grid_state, \
    state_with_sorrounding_objects, pg_map_api


def test_calculateLongitudinalDifferences(state_with_sorrounding_objects, behavioral_grid_state):

    target_map_states = [obj.map_state for obj in state_with_sorrounding_objects.dynamic_objects]

    longitudinal_distances = behavioral_grid_state.calculate_longitudinal_differences(target_map_states)

    for i, map_state in enumerate(target_map_states):
        rel_lane = RelativeLane(map_state.lane_id - behavioral_grid_state.ego_state.map_state.lane_id)
        assert longitudinal_distances[i] == map_state.lane_fstate[FS_SX] - behavioral_grid_state.projected_ego_fstates[rel_lane][FS_SX]
