import itertools
from abc import abstractmethod
from collections import defaultdict
from logging import Logger
from typing import List, Optional, Type

import numpy as np

from decision_making.src.exceptions import raises
from decision_making.src.global_constants import LON_ACC_LIMITS, LAT_ACC_LIMITS, VELOCITY_LIMITS, \
    BP_JERK_S_JERK_D_TIME_WEIGHTS
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.data_objects import ActionSpec
from decision_making.src.planning.behavioral.data_objects import AggressivenessLevel, \
    ActionRecipe
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
from decision_making.src.planning.types import FS_SV, FS_SX, FS_SA, FS_DX, FS_DV, FS_DA, \
    FrenetState2D, Limits, LIMIT_MIN, LIMIT_MAX
from decision_making.src.planning.utils.optimal_control.poly1d import Poly1D


class ActionSpace:
    def __init__(self, logger: Logger, recipes: List[ActionRecipe], recipe_filtering: Optional[RecipeFiltering] = None):
        """
        Abstract class for Action-Space implementations. Implementations should include actions enumeration, filtering
         and specification.
        :param logger: dedicated logger implementation
        :param recipes: list of recipes that define the scope of an ActionSpace implementation
        :param recipe_filtering: RecipeFiltering object that holds the logic for filtering recipes
        """
        self.logger = logger
        self._recipes = recipes
        self._recipe_filtering = recipe_filtering or RecipeFiltering()

    @property
    def action_space_size(self) -> int:
        return len(self._recipes)

    @property
    def recipes(self) -> List[ActionRecipe]:
        return self._recipes

    @property
    @abstractmethod
    def recipe_classes(self) -> List[Type]:
        pass

    def filter_recipe(self, recipe: ActionRecipe, behavioral_state: BehavioralState) -> bool:
        return self._recipe_filtering.filter_recipe(recipe, behavioral_state)

    def filter_recipes(self, action_recipes: List[ActionRecipe], behavioral_state: BehavioralState):
        """"""
        return self._recipe_filtering.filter_recipes(action_recipes, behavioral_state)

    @abstractmethod
    def specify_goal(self, action_recipe: ActionRecipe, behavioral_state: BehavioralGridState) -> Optional[ActionSpec]:
        """
        This method's purpose is to specify the enumerated actions that the agent can take.
        Each semantic action (action_recipe) is translated to a trajectory of the agent.
        The trajectory specification is created towards a target object in given cell in case of Dynamic action,
        and towards a certain lane in case of Static action, considering ego state.
        Internally, the reference route here is the RHS of the road, and the ActionSpec is specified with respect to it.
        :param action_recipe: an enumerated semantic action [ActionRecipe].
        :param behavioral_state: Frenet state of ego at initial point
        :return: semantic action specification [ActionSpec] or [None] if recipe can't be specified.
        """
        pass

    @abstractmethod
    def specify_goals(self, action_recipes: List[ActionRecipe], behavioral_state: BehavioralGridState) -> List[Optional[ActionSpec]]:
        pass

    @staticmethod
    def find_optimum_planning_time(T_vals: np.ndarray, poly_coefs_s: np.ndarray, poly_lib_s: Poly1D,
                                   poly_coefs_d: np.ndarray, poly_lib_d: Poly1D, agg_level: AggressivenessLevel):
        """
        Given planning horizons, lateral and longitudinal polynomials and aggressiveness level, this method finds the
        optimal time w.r.t cost defined by aggressiveness-level-dependent weights.
        :param T_vals: np.ndarray of planning horizons among which the action time specification will be taken.
        :param poly_coefs_s: coefficients of longitudinal polynomial [np.ndarray]
        :param poly_lib_s: library of type [Poly1D] for kinematic calculations.
        :param poly_coefs_d: coefficients of lateral polynomial [np.ndarray]
        :param poly_lib_d: library of type [Poly1D] for kinematic calculations.
        :param agg_level: [AggressivenessLevel]
        :return: a tuple of (optimum time horizon, whether this time horizon meets acceleration and velocity constraints)
        """
        jerk_s = poly_lib_s.cumulative_jerk(poly_coefs_s, T_vals)
        jerk_d = poly_lib_d.cumulative_jerk(poly_coefs_d, T_vals)

        cost = np.dot(np.c_[jerk_s, jerk_d, T_vals],
                      np.c_[BP_JERK_S_JERK_D_TIME_WEIGHTS[agg_level.value]])
        optimum_time_idx = np.argmin(cost)

        are_lon_acc_in_limits = poly_lib_s.are_accelerations_in_limits(poly_coefs_s, T_vals, LON_ACC_LIMITS)
        are_lat_acc_in_limits = poly_lib_d.are_accelerations_in_limits(poly_coefs_d, T_vals, LAT_ACC_LIMITS)
        are_vel_in_limits = poly_lib_s.are_velocities_in_limits(poly_coefs_s, T_vals, VELOCITY_LIMITS)

        optimum_time_satisfies_constraints = are_lon_acc_in_limits[optimum_time_idx] and \
                                             are_lat_acc_in_limits[optimum_time_idx] and \
                                             are_vel_in_limits[optimum_time_idx]

        return optimum_time_idx, optimum_time_satisfies_constraints

    @staticmethod
    def define_lon_constraints(repeat_factor: int, ego_init_fstate: FrenetState2D, desired_acc: float,
                               desired_vel: np.ndarray, desired_lon: np.ndarray = None):
        """
        Defines longitudinal constraints for Werling trajectory planning
        :param repeat_factor: number of planning horizons, determines the shape of returned tensor.
        :param ego_init_fstate: ego initial frenet-frame state
        :param desired_acc: desired acceleration when action is finished (for each planning horizon)
        :param desired_vel: desired velocity when action is finished(for each planning horizon)
        :param desired_lon: desired longitudinal position when action is finished (for each planning horizon, optional)
        :return: a tensor with the constraints of third-order dynamics in initial and terminal action phase.
        """
        if desired_lon is None:
            # Quartic polynomial constraints (no constraint on sT)
            constraints_s = np.repeat([[
                ego_init_fstate[FS_SX],
                ego_init_fstate[FS_SV],
                ego_init_fstate[FS_SA],
                desired_vel,  # desired velocity
                0.0  # zero acceleration at the end of action
            ]], repeats=repeat_factor, axis=0)

        else:
            # Quintic polynomial constraints
            constraints_s = np.c_[np.full(shape=repeat_factor, fill_value=ego_init_fstate[FS_SX]),
                                  np.full(shape=repeat_factor, fill_value=ego_init_fstate[FS_SV]),
                                  np.full(shape=repeat_factor, fill_value=ego_init_fstate[FS_SA]),
                                  desired_lon,
                                  desired_vel,
                                  np.full(shape=repeat_factor, fill_value=desired_acc)]

        return constraints_s

    @staticmethod
    def define_lat_constraints(repeat_factor: int, ego_init_fstate: FrenetState2D, desired_lat: float):
        """
        Defines lateral constraints for Werling trajectory planning
        :param repeat_factor: number of planning horizons, determines the shape of returned tensor.
        :param ego_init_fstate: ego initial frenet-frame state
        :param desired_lat: desired lateral position when action is finished (for each planning horizon)
        :return: a tensor with the constraints of third-order dynamics in initial and terminal action phase.
        """
        # Quintic polynomial constraints
        constraints_d = np.repeat([[
            ego_init_fstate[FS_DX],
            ego_init_fstate[FS_DV],
            ego_init_fstate[FS_DA],
            desired_lat,
            0.0,
            0.0
        ]], repeats=repeat_factor, axis=0)

        return constraints_d

    @staticmethod
    def find_real_roots(coef_matrix: np.ndarray, value_limits: Limits):
        """
        Given a matrix of polynomials coefficients, returns their Real roots within boundaries.
        :param coef_matrix: 2D numpy array [NxK] full with coefficients of N polynomials of degree (K-1)
        :param value_limits: Boundaries for desired roots to look for.
        :return: 2D numpy array [Nx(K-1)]
        """
        roots = np.apply_along_axis(np.roots, axis=-1, arr=coef_matrix)
        real_roots = np.real(roots)
        is_real = np.isclose(np.imag(roots), 0.0)
        is_in_limits = np.logical_and(real_roots >= value_limits[LIMIT_MIN], real_roots <= value_limits[LIMIT_MAX])
        real_roots[~np.logical_and(is_real,  is_in_limits)] = np.nan
        return real_roots


class ActionSpaceContainer(ActionSpace):
    def __init__(self, logger: Logger, action_spaces: List[ActionSpace]):
        super().__init__(logger, [])
        self._action_spaces = action_spaces

        self._recipe_handler = {recipe_class: aspace
                                for aspace in action_spaces
                                for recipe_class in aspace.recipe_classes}

    @property
    def action_space_size(self) -> int:
        return sum(aspace.action_space_size for aspace in self._action_spaces)

    @property
    def recipes(self) -> List[ActionRecipe]:
        return list(itertools.chain.from_iterable(aspace.recipes for aspace in self._action_spaces))

    @property
    def recipe_classes(self) -> List:
        return list(itertools.chain.from_iterable(aspace.recipe_classes for aspace in self._action_spaces))

    @raises(NotImplemented)
    def specify_goal(self, action_recipe: ActionRecipe, behavioral_state: BehavioralGridState) -> Optional[ActionSpec]:
        return self._recipe_handler[action_recipe.__class__].specify_goal(action_recipe, behavioral_state)

    def specify_goals(self, action_recipes: List[ActionRecipe], behavioral_state: BehavioralGridState) -> \
            List[Optional[ActionSpec]]:
        grouped_actions = defaultdict(list)
        grouped_idxs = defaultdict(list)
        for idx, recipe in enumerate(action_recipes):
            grouped_actions[recipe.__class__].append(recipe)
            grouped_idxs[recipe.__class__].append(idx)

        indexed_action_specs = list(itertools.chain.from_iterable(
            [zip(grouped_idxs[action_class], self._recipe_handler[action_class].specify_goals(action_list, behavioral_state))
             for action_class, action_list in grouped_actions.items()]))

        return [action for idx, action in sorted(indexed_action_specs, key=lambda idx_action: idx_action[0])]

    @raises(NotImplemented)
    def filter_recipe(self, action_recipe: ActionRecipe, behavioral_state: BehavioralGridState) -> bool:
        try:
            return self._recipe_handler[action_recipe.__class__].filter_recipe(action_recipe, behavioral_state)
        except Exception:
            raise NotImplemented('action_recipe %s could not be handled by current action spaces %s',
                                 action_recipe, str(self._action_spaces))

    # TODO: figure out how to remove the for loop for better efficiency and stay consistent with ordering
    @raises(NotImplemented)
    def filter_recipes(self, action_recipes: List[ActionRecipe], behavioral_state: BehavioralState):
        try:
            return [self._recipe_handler[action_recipe.__class__].filter_recipe(action_recipe, behavioral_state)
                    for action_recipe in action_recipes]
        except Exception:
            raise NotImplemented('an action_recipe could not be handled by current action spaces %s',
                                 str(self._action_spaces))
