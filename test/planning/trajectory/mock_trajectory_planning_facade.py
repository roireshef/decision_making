from logging import Logger

from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.messages.trajectory_plan_message import TrajectoryPlanMsg
from decision_making.src.messages.visualization.trajectory_visualization_message import TrajectoryVisualizationMsg
from decision_making.src.planning.trajectory.optimal_control.werling_planner import WerlingPlanner
from decision_making.src.planning.trajectory.trajectory_planning_facade import TrajectoryPlanningFacade
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy


class TrajectoryPlanningSimulationFacadeMock(TrajectoryPlanningFacade):
    """
    Sends periodic dummy trajectory message
    """
    def __init__(self, dds: DdsPubSub, logger: Logger, trajectory_msg: TrajectoryPlanMsg,
                 visualization_msg: TrajectoryVisualizationMsg):
        """
        :param dds: communication layer (DDS) instance
        :param logger: logger
        :param trajectory_msg: the trajectory message to publish periodically
        :param visualization_msg: the visualization message to publish periodically
        """

        planner = WerlingPlanner(logger)
        strategy_handlers = {TrajectoryPlanningStrategy.HIGHWAY: planner,
                             TrajectoryPlanningStrategy.PARKING: planner,
                             TrajectoryPlanningStrategy.TRAFFIC_JAM: planner}
        TrajectoryPlanningFacade.__init__(self, dds, logger, strategy_handlers)
        self._trajectory_msg = trajectory_msg
        self._visualization_msg = visualization_msg

    def _periodic_action_impl(self):
        """
        This mock sends the received messages from the init.
        """

        # publish results to the lower DM level
        self._publish_trajectory(self._trajectory_msg)

        # publish visualization/debug data
        self._publish_debug(self._visualization_msg)
