from decision_making.src.planning.behavioral.semantic_actions_policy import SemanticActionsPolicy


SEMANTIC_GRID_FRONT, SEMANTIC_GRID_ASIDE, SEMANTIC_GRID_BEHIND = 1, 0, -1
GRID_MID = 10

# The margin that we take from the front/read of the vehicle to define the front/rear partitions
SEMANTIC_OCCUPANCY_GRID_PARTITIONS_MARGIN_FROM_EGO = 1


class NovDemoPolicy(SemanticActionsPolicy):

    def _eval_actions(self, state: State, semantic_actions: List[SemanticAction],
                      actions_spec: List[SemanticActionSpec]) -> np.ndarray:
        """
        Evaluate the generated actions using the full state.
        Gets a list of actions to evaluate so and returns a vector representing their costs.
        A set of actions is provided, enabling assessing them dependently.
        Note: the semantic actions were generated using the behavioral state which isn't necessarily captures
         all relevant details in the scene. Therefore the evaluation is done using the full state.
        :param state: world state
        :param semantic_actions: semantic actions list
        :param actions_spec: specifications of semantic actions
        :return: numpy array of costs of semantic actions
        """
        costs = np.zeros(len(semantic_actions))
        follow_current_lane_ind = [i for i, action in enumerate(semantic_actions)
                                   if action.cell[SEMANTIC_CELL_LANE] == 0]
        follow_left_lane_ind = [i for i, action in enumerate(semantic_actions)
                                if action.cell[SEMANTIC_CELL_LANE] == 1]
        follow_right_lane_ind = [i for i, action in enumerate(semantic_actions)
                                 if action.cell[SEMANTIC_CELL_LANE] == -1]

        if follow_current_lane_ind is None:
            raise error

        move_right = follow_right_lane_ind is not None and \
                     max_velocity - actions_spec[follow_right_lane_ind].v < MIN_OVERTAKE_VEL and \
                     max_velocity - actions_spec[follow_current_lane_ind].v < MIN_OVERTAKE_VEL

        move_left = follow_left_lane_ind is not None and \
                    max_velocity - actions_spec[follow_current_lane_ind].v >= MIN_OVERTAKE_VEL and \
                    actions_spec[follow_left_lane_ind].v - actions_spec[follow_current_lane_ind].v >= MIN_OVERTAKE_VEL

        if move_right:
            costs[follow_right_lane_ind] = 1.
        elif move_left:
            costs[follow_left_lane_ind] = 1.
        else:
            costs[follow_current_lane_ind] = 1.

        return costs
