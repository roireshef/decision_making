from logging import Logger

from common_data.src.communication.pubsub.pubsub import PubSub
from decision_making.src.messages.trajectory_plan_message import TrajectoryPlanMsg
from decision_making.src.messages.visualization.trajectory_visualization_message import TrajectoryVisualizationMsg
from decision_making.src.planning.trajectory.trajectory_planner import SamplableTrajectory
from decision_making.src.planning.trajectory.trajectory_planning_facade import TrajectoryPlanningFacade
from decision_making.src.prediction.predictor import Predictor


class TrajectoryPlanningFacadeMock(TrajectoryPlanningFacade):
    """
    Sends periodic dummy trajectory message
    """

    def __init__(self, pubsub: PubSub, logger: Logger, trajectory_msg: TrajectoryPlanMsg,
                 visualization_msg: TrajectoryVisualizationMsg, last_trajectory: SamplableTrajectory = None):
        """
        :param pubsub: communication layer (DDS/LCM/...) instance
        :param logger: logger
        :param trajectory_msg: the trajectory message to publish periodically
        :param visualization_msg: the visualization message to publish periodically
        :param last_trajectory: only for unit-test purposes. same logic as TrajectoryPlanningFacade._last_trajectory
        """
        TrajectoryPlanningFacade.__init__(self, pubsub, logger, None, None)
        self._trajectory_msg = trajectory_msg
        self._visualization_msg = visualization_msg
        self._last_trajectory = last_trajectory

    def _validate_strategy_handlers(self) -> None:
        pass

    def _periodic_action_impl(self):
        """
        This mock sends the received messages from the init.
        """
        # publish results to the lower DM level
        self._publish_trajectory(self._trajectory_msg)

        # publish visualization/debug data
        self._publish_debug(self._visualization_msg)
