from abc import abstractmethod
from itertools import compress
from typing import Dict
from typing import List

import numpy as np
from decision_making.src.planning.types import BoolArray
from gym.spaces import Discrete
from decision_making.src.rl_agent.environments.action_space.common.data_objects import RLActionRecipe
from decision_making.src.rl_agent.environments.action_space.common.data_objects import RLActionSpec
from decision_making.src.rl_agent.environments.action_space.filters.action_spec_filter import ActionSpecFilter
from decision_making.src.rl_agent.environments.sim_commands import SimCommand
from decision_making.src.rl_agent.environments.state_space.common.data_objects import EgoCentricState


class ActionSpaceAdapter:
    """
    Wraps the action space and provides methods to convert action selections to trajectories.
    """
    RECIPE_CLS = RLActionRecipe

    def __init__(self, action_space_params: Dict, action_recipes: List[RECIPE_CLS]):
        self._params = action_space_params
        self._action_recipes = action_recipes

    @property
    def action_recipes(self) -> List[RECIPE_CLS]:
        return self._action_recipes

    @property
    def action_space(self):
        """
        :return: a gym.spaces.Space to use in the environment
        """
        return Discrete(n=len(self.action_recipes))

    @property
    def filters(self) -> List[ActionSpecFilter]:
        return self._params['FILTERS']

    def specify_action(self, state: EgoCentricState, action_recipe: RECIPE_CLS) -> RLActionSpec:
        return self.specify_actions(state, [action_recipe])[0]

    @abstractmethod
    def specify_actions(self, state: EgoCentricState, action_recipes: List[RECIPE_CLS]) -> List[RLActionSpec]:
        """
        A specify method
        :param state: a state to use for specifying actions (ego and actors)
        :param action_recipes: a list of action recipes
        :return:
        """
        pass

    def get_action_mask(self, state: EgoCentricState) -> (BoolArray, np.array):
        """
        Runs all filters and returns their results as a mask vector, as well as for each action - the id of the filter
        who (was the first to) invalidated it. To see a mapping of all filters by IDs - run filters_bank.py
        :param state: a state to use for specifying actions (ego and actors)
        :return: tuple of (1d numpy bool array for actions mask; 1d numpy array of the ID of the filter which
        invalidated each action, or -1 if action is valid)
        """
        action_specs = self.specify_actions(state, self.action_recipes)
        filtering_map = -1 * np.ones(len(action_specs))
        mask = np.full(shape=len(action_specs), fill_value=True, dtype=np.bool)

        for filter_idx, action_spec_filter in enumerate(self.filters):
            if ~np.any(mask):
                break

            # list of only valid action specs
            valid_action_specs = list(compress(action_specs, mask))

            # a mask only on the valid action specs
            current_mask = action_spec_filter.filter(valid_action_specs, state)

            # Stores the id of the filter who failed each action
            current_filter_map = filtering_map[mask]
            current_filter_map[~current_mask] = action_spec_filter.id
            filtering_map[mask] = current_filter_map

            # use the reduced mask to update the original mask (that contains all initial actions specs given)
            mask[mask] = current_mask

        return mask, filtering_map

    @abstractmethod
    def get_commands(self, state: EgoCentricState, action_id: int) -> (List[SimCommand], Dict):
        """
        Maps an action id (from the action space) into simulation commands (accelerations for sumo)
        :param state: EgoCentricState object that represents the environment relative to ego
        :param action_id: action index out of the provided action space
        :return: A tuple of: (list of simulation commands (each for each sim step), debug data as a dict)
        """
        pass
