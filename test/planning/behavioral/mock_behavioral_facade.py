from logging import Logger
from typing import Optional

from common_data.src.communication.pubsub.pubsub import PubSub
from decision_making.src.global_constants import NEGLIGIBLE_DISPOSITION_LON, NEGLIGIBLE_DISPOSITION_LAT
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.behavioral_facade import BehavioralFacade
from decision_making.src.planning.trajectory.trajectory_planning_facade import TrajectoryPlanningFacade
import numpy as np

class BehavioralFacadeMock(BehavioralFacade):
    """
    Operate according to to policy with an empty dummy behavioral state
    """
    def __init__(self, pubsub: PubSub, logger: Logger, trajectory_params: TrajectoryParams,
                 visualization_msg: BehavioralVisualizationMsg):
        """
        :param pubsub: communication layer (DDS/LCM/...) instance
        :param logger: logger
        :param trajectory_params: the trajectory params message to publish periodically
        :param visualization_msg: the visualization message to publish periodically
        """
        super().__init__(pubsub=pubsub, logger=logger, policy=None)
        self._trajectory_params = trajectory_params
        self._visualization_msg = visualization_msg
        self._triggered = False

    def _periodic_action_impl(self):
        """
        Publishes the received messages initialized in init
        :return: void
        """
        state = self._get_current_state()
        current_pos = np.array([state.ego_state.x, state.ego_state.y])

        if not self._triggered and np.all(np.abs(current_pos - self._trigger_pos) <
                                          np.array([NEGLIGIBLE_DISPOSITION_LON, NEGLIGIBLE_DISPOSITION_LAT])):
            self._triggered = True

            # NOTE THAT TIMESTAMP IS UPDATED HERE !
            self._trajectory_params.time += state.ego_state.timestamp_in_sec

        if self._triggered:
            self._publish_results(self._trajectory_params)
            self._publish_visualization(self._visualization_msg)
