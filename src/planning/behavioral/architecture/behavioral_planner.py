import numpy as np

from decision_making.src.planning.behavioral.architecture.action_evaluator import ActionEvaluator
from decision_making.src.planning.behavioral.architecture.action_space import ActionSpace
from decision_making.src.planning.behavioral.architecture.action_validator import ActionValidator
from decision_making.src.planning.behavioral.architecture.value_approximator import ValueApproximator
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State


class BehavioralPlanner:
    def __init__(self, action_space: ActionSpace, action_evaluator: ActionEvaluator,
                 action_validator: ActionValidator, value_approximator: ValueApproximator,
                 predictor: Predictor):
        self.action_space = action_space
        self.action_evaluator = action_evaluator
        self.action_validator = action_validator
        self.value_approximator = value_approximator
        self.predictor = predictor


class CostBasedBehavioralPlanner(BehavioralPlanner):
    def __init__(self, action_space: ActionSpace, action_evaluator: ActionEvaluator, action_validator: ActionValidator,
                 value_approximator: ValueApproximator, predictor: Predictor):
        super().__init__(action_space, action_evaluator, action_validator, value_approximator, predictor)

    def plan(self, state: State):
        action_recipes = self.action_space.generate_recipes()
        action_specs = dict(zip(action_recipes, self.action_space.specify_goals(action_recipes, state)))

        action_filters = {recipe: self.action_validator.validate(goal) for recipe, goal in action_specs.items()}

        action_costs = np.full(shape=self.action_space.action_space_size, fill_value=np.inf)
        action_costs[action_filters] = [self.action_evaluator.evaluate(state, goal) +
                                        self.value_approximator.evaluate(self.predictor.predict(state, goal))
                                        for recipe, goal in action_specs]

        cost_sorted_indices = action_costs.argsort()

        return action_specs[cost_sorted_indices], action_costs[cost_sorted_indices]











