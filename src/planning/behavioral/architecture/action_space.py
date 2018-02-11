from abc import abstractmethod, abstractproperty
from typing import List

import numpy as np

from decision_making.src.planning.behavioral.architecture.constants import VELOCITY_STEP, MAX_VELOCITY, MIN_VELOCITY
from decision_making.src.planning.behavioral.architecture.data_objects import ActionSpec, StaticActionRecipe, \
    DynamicActionRecipe
from decision_making.src.planning.behavioral.architecture.data_objects import RelativeLane, AggressivenessLevel, \
    RelativeLongitudinalPosition, ActionRecipe
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State


class ActionSpace:
    @abstractproperty
    def action_space_size(self) -> int:
        pass

    @abstractmethod
    def generate_recipes(self) -> List[ActionRecipe]:
        """enumerate the whole action space (independent of state)"""
        pass

    @abstractmethod
    def specify_goals(self, action_recipes: List[ActionRecipe], state: State) -> List[ActionSpec]:
        """"""
        pass

    def generate_actions(self, state):
        recipes = self.generate_recipes()
        return dict(zip(recipes, self.specify_goals(recipes, state)))


class StaticActionSpace(ActionSpace):
    @property
    def velocities(self):
        return np.arange(MIN_VELOCITY, MAX_VELOCITY + np.finfo(np.float16).eps, VELOCITY_STEP)

    @property
    def action_space_size(self):
        return RelativeLane.__len__() * AggressivenessLevel.__len__() * self.velocities.__len__()

    def generate_recipes(self) -> List[ActionRecipe]:
        pass

    def specify_goals(self, action_recipes: List[ActionRecipe], state: State) -> List[ActionSpec]:
        pass


class DynamicActionSpace(ActionSpace):
    def __init__(self, predictor: Predictor):
        self.predictor = predictor

    @property
    def action_space_size(self):
        # 2 = FOLLOW_VEHICLE, TAKE_OVER_VEHICLE
        return RelativeLane.__len__() * AggressivenessLevel.__len__() * RelativeLongitudinalPosition.__len__() * 2

    def generate_recipes(self) -> List[ActionRecipe]:
        pass

    def specify_goals(self, action_recipes: List[ActionRecipe], state: State) -> List[ActionSpec]:
        # generate dict of {cell (from ActionRecipes given): (relevant) DynamicObject}
        # transform to dict of {cell: (predictions on road) List[...?]}
        # call dedicated method for each recipe-type.
        pass


class CombinedActionSpace(StaticActionSpace, DynamicActionSpace):
    def __init__(self, predictor: Predictor):
        DynamicActionSpace.__init__(self, predictor)

    @property
    def action_space_size(self):
        return super(StaticActionSpace, self).action_space_size + super(DynamicActionSpace, self).action_space_size

    def generate_recipes(self):
        static_recipes = super(StaticActionSpace, self).generate_recipes()
        dynamic_recipes = super(DynamicActionSpace, self).generate_recipes()
        return static_recipes + dynamic_recipes

    def specify_goals(self, action_recipes: List[ActionRecipe], state: State) -> List[ActionSpec]:
        static_recipes = [a for a in action_recipes if isinstance(a, StaticActionRecipe)]
        dynamic_recipes = [a for a in action_recipes if isinstance(a, DynamicActionRecipe)]

        static_goals = super(StaticActionSpace, self).specify_goals(static_recipes, state)
        dynamic_goals = super(DynamicActionSpace, self).specify_goals(dynamic_recipes, state)

        return static_goals + dynamic_goals
