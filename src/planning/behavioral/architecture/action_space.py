from typing import List

import numpy as np

from decision_making.src.planning.behavioral.architecture.data_objects import ActionSpec
from decision_making.src.planning.behavioral.architecture.data_objects import RelativeLane, AggressivenessLevel, \
    RelativeLongitudinalPosition, ActionRecipe
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State


class ActionSpace:
    def __init__(self, predictor: Predictor):
        self.predictor = predictor

    @property
    def velocities(self):
        return np.arange(20, 120, 10) / 3.6

    @property
    def action_space_size(self):
        return RelativeLane.__len__() * AggressivenessLevel.__len__() * (
            len(self.velocities) +                      # FOLLOW_LANE
            2 * RelativeLongitudinalPosition.__len__()  # FOLLOW_VEHICLE, TAKE_OVER_VEHICLE
        )

    def generate_actions(self, state):
        recipes = self.generate_recipes()
        return dict(zip(recipes, self.specify_goals(recipes, state)))

    def generate_recipes(self) -> List[ActionRecipe]:
        """enumerate the whole action space (independent of state)"""
        pass

    def specify_goals(self, action_recipes: List[ActionRecipe], state: State) -> List[ActionSpec]:
        """"""
        # generate dict of {cell (from ActionRecipes given): (relevant) DynamicObject}
        # transform to dict of {cell: (predictions on road) List[...?]}
        # call dedicated method for each recipe-type.
        pass
