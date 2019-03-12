import six
from abc import abstractmethod, ABCMeta
from logging import Logger
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.state.state import State


@six.add_metaclass(ABCMeta)
class BehavioralPlannerBase:
    def __init__(self, predictor: EgoAwarePredictor, logger: Logger):
        self.predictor = predictor
        self.logger = logger

    @abstractmethod
    def plan(self, state: State, nav_plan: NavigationPlanMsg):
        """
        Given current state and navigation plan, plans the next semantic action to be carried away. This method makes
        use of Planner components such as Evaluator,Validator and Predictor for enumerating, specifying
        and evaluating actions. Its output will be further handled and used to create a trajectory in Trajectory Planner
        and has the form of TrajectoryParams, which includes the reference route, target time, target state to be in,
        cost params and strategy.
        :param state: the current world state
        :param nav_plan:
        :return: a tuple: (TrajectoryParams for TP,BehavioralVisualizationMsg for e.g. VizTool)
        """
        pass
