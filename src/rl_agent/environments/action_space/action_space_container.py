import itertools
from typing import List, Dict, Union, Type, Optional

import numpy as np
from gym.spaces import Discrete
from decision_making.src.rl_agent.environments.action_space.action_space_adapter import ActionSpaceAdapter
from decision_making.src.rl_agent.environments.action_space.common.data_objects import RLActionRecipe
from decision_making.src.rl_agent.environments.action_space.common.data_objects import RLActionSpec
from decision_making.src.rl_agent.environments.sim_commands import SimCommand
from decision_making.src.rl_agent.environments.state_space.common.data_objects import EgoCentricState
from decision_making.src.rl_agent.utils.numpy_utils import NumpyUtils
from decision_making.src.rl_agent.utils.primitive_utils import VectorizedDict, DictUtils
from ray.tune.registry import RLLIB_PREPROCESSOR, _global_registry as ray_registry


class ActionSpaceAdapterContainer(ActionSpaceAdapter):
    def __init__(self, action_space_params: Dict, common_params: Optional[Dict] = None):
        """
        This a wrapper class for the purpose of combining one or more discrete action space adapters.
        As such, it exposes the same interface and takes care of all the aggregations of all methods results.
        :param action_space_params: dictionary mapping action spaces names to their parameters
        """
        action_spaces = [
            self._get_action_space_adapter(as_name)(DictUtils.recursive_merge_with(as_params, common_params or {}))
            for as_name, as_params in action_space_params.items()
        ]
        assert all(type(aspace.action_space) == Discrete for aspace in action_spaces), \
            "current implementation includes only discrete action spaces"

        super().__init__(action_space_params,
                         list(itertools.chain.from_iterable(aspace.action_recipes for aspace in action_spaces)))

        self._action_spaces = action_spaces
        self._action_recipes_to_ids = VectorizedDict(map(reversed, enumerate(self._action_recipes)))
        self._action_ids_to_recipes = VectorizedDict(enumerate(self._action_recipes))

        # Caches the number of actions in each action space & the offset (sum of all actions in previous action spaces)
        self._action_spaces_len = np.array([aspace.action_space.n for aspace in self._action_spaces])
        self._action_spaces_id_offsets = np.insert(np.cumsum(self._action_spaces_len), 0, 0., axis=0)

    @property
    def action_space(self):
        # Actions are combined so their number is sum of all combined action spaces
        return Discrete(n=np.sum(self._action_spaces_len))

    def specify_actions(self, state: EgoCentricState, action_recipes: List[RLActionRecipe]) -> List[RLActionSpec]:
        action_ids = list(self._action_recipes_to_ids[action_recipes])
        action_ids_arr = np.array(action_ids)

        # Get a list of action space indices corresponding to each action in given list
        action_space_idxs = self._map_action_id_to_action_space_idx(action_ids)

        # Iterate over all available action spaces, look for relevant actions in the list given, and specify if exist
        result_sorted = np.empty_like(action_ids, dtype=object)

        for aspace_idx in set(action_space_idxs):
            # slice the action ids relevant to this action-space (by aspace_idx)
            external_action_ids = action_ids_arr[action_space_idxs == aspace_idx]

            # create a list of corresponding recipes and "specify" them into action-specs
            aspace_recipes = self._action_ids_to_recipes[external_action_ids]
            action_specs = self._action_spaces[aspace_idx].specify_actions(state, aspace_recipes)

            # write to result_sorted array in the relevant indices (by aspace_idx)
            np.putmask(result_sorted, action_space_idxs == aspace_idx, NumpyUtils.from_list_of_tuples(action_specs))

        return list(result_sorted)

    def get_action_mask(self, state: EgoCentricState) -> (np.ndarray, np.ndarray):
        # Concatenates the "mask arrays" and "filters maps" (which filter invalidated an action) for all actions
        masks_list, filter_mappings_list = zip(*(aspace.get_action_mask(state) for aspace in self._action_spaces))
        return np.concatenate(masks_list), np.concatenate(filter_mappings_list)

    def get_commands(self, state: EgoCentricState, action_id: int) -> (List[SimCommand], Dict):
        # Per the given action ID, this looks what is the correct action space to use and then offsets the ID to get
        # the correct ID in this action space
        action_space_idx = self._map_action_id_to_action_space_idx([action_id])[0]
        action_space_action_id = action_id - self._action_spaces_id_offsets[action_space_idx]
        return self._action_spaces[action_space_idx].get_commands(state, action_space_action_id)

    def _map_action_id_to_action_space_idx(self, action_ids: Union[List[int], np.ndarray]) -> np.ndarray:
        # this usage of np.searchsorted does interpolation on self._action_spaces_id_offsets, so 'right' (and -1) must
        # be used to avoid wrong results
        aspace_idx = np.searchsorted(self._action_spaces_id_offsets, action_ids, side='right') - 1
        return aspace_idx

    @staticmethod
    def _get_action_space_adapter(name: str) -> Type[ActionSpaceAdapter]:
        """ pulls back an action space adapter class from registry, by name """
        return ray_registry.get(RLLIB_PREPROCESSOR, name)
