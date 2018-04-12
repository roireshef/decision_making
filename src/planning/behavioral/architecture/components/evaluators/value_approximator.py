from logging import Logger
from decision_making.src.planning.behavioral.policies.semantic_actions_grid_state import SemanticActionsGridState


class ValueApproximator:
    def __init__(self, logger: Logger):
        self.logger = logger

    def evaluate_state(self, behavioral_state: SemanticActionsGridState) -> float:
        pass
