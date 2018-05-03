import itertools
from abc import abstractmethod
from logging import Logger
from typing import List, Optional

import numpy as np

from decision_making.src.exceptions import raises
from decision_making.src.global_constants import LON_ACC_LIMITS, LAT_ACC_LIMITS, VELOCITY_LIMITS, \
    BP_JERK_S_JERK_D_TIME_WEIGHTS
from decision_making.src.planning.behavioral.architecture.components.filtering.recipe_filtering import RecipeFiltering
from decision_making.src.planning.behavioral.architecture.data_objects import ActionSpec
from decision_making.src.planning.behavioral.architecture.data_objects import AggressivenessLevel, \
    ActionRecipe
from decision_making.src.planning.behavioral.architecture.semantic_behavioral_grid_state import \
    SemanticBehavioralGridState
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.trajectory.optimal_control.optimal_control_utils import Poly1D
from decision_making.src.planning.types import FS_SV, FS_SX, FS_SA, FS_DX, FS_DV, FS_DA, \
    FrenetState2D


class ActionSpace:
    def __init__(self, logger: Logger, recipes: List[ActionRecipe], recipe_filtering: RecipeFiltering = None):
        """
        Abstract class for Action-Space implementations. Implementations should include actions enumeration, filtering
         and specification.
        :param logger: dedicated logger implementation
        :param recipes: list of recipes that define the scope of an ActionSpace implementation
        :param recipe_filtering: RecipeFiltering object that holds the logic for filtering recipes
        """
        self.logger = logger
        self._recipes = recipes
        self.recipe_filtering = recipe_filtering or RecipeFiltering()

    @property
    def action_space_size(self) -> int:
        return len(self._recipes)

    @property
    def recipes(self) -> List[ActionRecipe]:
        return self._recipes

    def filter_recipe(self, recipe: ActionRecipe, behavioral_state: BehavioralState) -> bool:
        return self.recipe_filtering.filter_recipe(recipe, behavioral_state)

    def filter_recipes(self, action_recipes: List[ActionRecipe], behavioral_state: BehavioralState):
        """"""
        return self.recipe_filtering.filter_recipes(action_recipes, behavioral_state)

    @abstractmethod
    def specify_goal(self, action_recipe: ActionRecipe, behavioral_state: BehavioralState) -> Optional[ActionSpec]:
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


class ActionSpaceContainer:
    def __init__(self, logger: Logger, action_spaces: List[ActionSpace]):
        self._action_spaces = action_spaces
        self.logger = logger

        self._recipe_handler = {}
        for aspace in action_spaces:
            for recipe in aspace.recipes:
                self._recipe_handler[recipe] = aspace

    @property
    def action_space_size(self) -> int:
        return sum(aspace.action_space_size for aspace in self._action_spaces)

    @property
    def recipes(self) -> List[ActionRecipe]:
        return list(itertools.chain.from_iterable(aspace.recipes for aspace in self._action_spaces))

    @raises(NotImplemented)
    def specify_goal(self, action_recipe: ActionRecipe, behavioral_state: SemanticBehavioralGridState) -> ActionSpec:
        try:
            return self._recipe_handler[action_recipe].specify_goal(action_recipe, behavioral_state)
        except Exception:
            raise NotImplemented('action_recipe %s could not be handled by current action spaces %s',
                                 action_recipe, str(self._action_spaces))

    @raises(NotImplemented)
    def filter_recipe(self, action_recipe: ActionRecipe, behavioral_state: SemanticBehavioralGridState) -> bool:
        try:
            return self._recipe_handler[action_recipe].filter_recipe(action_recipe, behavioral_state)
        except Exception:
            raise NotImplemented('action_recipe %s could not be handled by current action spaces %s',
                                 action_recipe, str(self._action_spaces))

    def filter_recipes(self, action_recipes: List[ActionRecipe], behavioral_state: BehavioralState):
        """"""
        try:
            return [self._recipe_handler[action_recipe].filter_recipe(action_recipe, behavioral_state) for action_recipe
                    in action_recipes]
        except Exception:
            raise NotImplemented('an action_recipe could not be handled by current action spaces %s',
                                 str(self._action_spaces))
