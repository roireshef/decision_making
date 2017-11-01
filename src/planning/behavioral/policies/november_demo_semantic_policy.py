from typing import List

from decision_making.src.planning.behavioral.semantic_actions_policy import SemanticActionsPolicy, SemanticAction, \
    SemanticActionType, SemanticBehavioralState

SEMANTIC_GRID_FRONT, SEMANTIC_GRID_ASIDE, SEMANTIC_GRID_BEHIND = 1, 0, -1
GRID_MID = 10

# The margin that we take from the front/read of the vehicle to define the front/rear partitions
SEMANTIC_OCCUPANCY_GRID_PARTITIONS_MARGIN_FROM_EGO = 1


class NovDemoPolicy(SemanticActionsPolicy):

    def _enumerate_actions(self, behavioral_state: SemanticBehavioralState) -> List[SemanticAction]:
        """
        Enumerate the list of possible semantic actions to be generated.
        :param behavioral_state:
        :return:
        """

        semantic_actions: List[SemanticAction] = list()

        # Generate actions towards each of the cells in front of ego
        for relative_lane_key in [-1, 0, 1]:
            for longitudinal_key in [SEMANTIC_GRID_FRONT]:
                semantic_cell = (relative_lane_key, longitudinal_key)
                if semantic_cell in behavioral_state.road_occupancy_grid:
                    # Select first (closest) object in cell
                    target_obj = behavioral_state.road_occupancy_grid[semantic_cell][0]
                else:
                    # There are no objects in cell
                    target_obj = None

                semantic_action = SemanticAction(cell=semantic_cell, target_obj=target_obj,
                                                 action_type=SemanticActionType.FOLLOW)

                semantic_actions.append(semantic_action)

        return semantic_actions

