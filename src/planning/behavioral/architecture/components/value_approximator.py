from logging import Logger

from decision_making.src.planning.behavioral.architecture.semantic_behavioral_grid_state import \
    SemanticBehavioralGridState


class ValueApproximator:
    def __init__(self, logger: Logger):
        self.logger = logger

    def evaluate_state(self, behavioral_state: SemanticBehavioralGridState) -> float:
        state_val = 0.0
        self.logger.debug("state %s value is: %d", str(behavioral_state), state_val)
        return state_val
