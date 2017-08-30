from decision_making.src.messages.trajectory_plan_message import TrajectoryPlanMsg
from decision_making.src.planning.trajectory.optimal_control.werling_planner import WerlingPlanner
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.trajectory.trajectory_planning_facade import TrajectoryPlanningFacade
from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from logging import Logger
import numpy as np


class TrajectoryPlanningSimulationFacadeMock(TrajectoryPlanningFacade):
    """
    Sends periodic dummy trajectory message
    """
    def __init__(self, dds: DdsPubSub, logger: Logger):
        planner = WerlingPlanner(logger)
        strategy_handlers = {TrajectoryPlanningStrategy.HIGHWAY: planner,
                             TrajectoryPlanningStrategy.PARKING: planner,
                             TrajectoryPlanningStrategy.TRAFFIC_JAM: planner}
        TrajectoryPlanningFacade.__init__(self, dds, logger, strategy_handlers)

    def _periodic_action_impl(self):
        """
        will execute planning with using the implementation. This mock sends a dummy message
        """

        trajectory = np.array(
            [[1.0, 0.0, 0.0, 0.0], [2.0, -0.33, 0.0, 0.0], [3.0, -0.66, 0.0, 0.0], [4.0, -1.0, 0.0, 0.0],
             [5.0, -1.33, 0.0, 0.0], [6.0, -1.66, 0.0, 0.0], [7.0, -2.0, 0.0, 0.0], [8.0, -2.0, 0.0, 0.0],
             [9.0, -2.0, 0.0, 0.0], [10.0, -2.0, 0.0, 0.0], [11.0, -2.0, 0.0, 0.0]])
        ref_route = np.array(
            [[1.0, -2.0, 0.0], [2.0, -2.0, 0.0], [3.0, -2.0, 0.0], [4.0, -2.0, 0.0], [5.0, -2.0, 0.0],
             [6.0, -2.0, 0.0],
             [7.0, -2.0, 0.0], [8.0, -2.0, 0.0], [9.0, -2.0, 0.0], [10.0, -2.0, 0.0], [11.0, -2.0, 0.0],
             [12.0, -2.0, 0.0], [13.0, -2.0, 0.0], [14.0, -2.0, 0.0], [15.0, -2.0, 0.0], [16.0, -2.0, 0.0]])

        # publish results to the lower DM level
        self._publish_trajectory(TrajectoryPlanMsg(trajectory=trajectory, reference_route=ref_route, current_speed=5.0))

        # publish visualization/debug data
        #self.__publish_debug(debug_results)