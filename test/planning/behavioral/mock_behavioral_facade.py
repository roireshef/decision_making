import traceback
from logging import Logger
from typing import Optional

import numpy as np

from decision_making.src.infra.pubsub import PubSub
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.behavioral_planning_facade import BehavioralPlanningFacade
from decision_making.src.planning.types import CartesianPoint2D
from decision_making.test.constants import BP_NEGLIGIBLE_DISPOSITION_LON, BP_NEGLIGIBLE_DISPOSITION_LAT


class BehavioralFacadeMock(BehavioralPlanningFacade):
    """
    Operate according to to policy with an empty dummy behavioral state
    """
    def __init__(self, pubsub: PubSub, logger: Logger, trigger_pos: Optional[CartesianPoint2D],
                 trajectory_params: TrajectoryParams, visualization_msg: BehavioralVisualizationMsg):
        """
        :param pubsub: communication layer (DDS/LCM/...) instance
        :param logger: logger
        :param trigger_pos: the position that triggers the first output, None if there is no need in triggering mechanism
        :param trajectory_params: the trajectory params message to publish periodically
        :param visualization_msg: the visualization message to publish periodically
        """
        super().__init__(pubsub=PubSub, logger=logger, behavioral_planner=None,
                         last_trajectory=None)
        self._trajectory_params = trajectory_params
        self._visualization_msg = visualization_msg

        self._trigger_pos = trigger_pos
        if self._trigger_pos is None:
            self._triggered = True
        else:
            self._triggered = False

    def _periodic_action_impl(self):
        """
        Publishes the received messages initialized in init
        :return: void
        """
        try:
            state = self._get_current_state()
            current_pos = np.array([state.ego_state.x, state.ego_state.y])

            if not self._triggered and np.all(np.abs(current_pos - self._trigger_pos) <
                                              np.array([BP_NEGLIGIBLE_DISPOSITION_LON, BP_NEGLIGIBLE_DISPOSITION_LAT])):
                self._triggered = True

                # NOTE THAT TIMESTAMP IS UPDATED HERE !
                self._trajectory_params.time += state.ego_state.timestamp_in_sec

            if self._triggered:
                self._publish_results(self._trajectory_params)
                self._publish_visualization(self._visualization_msg)
            else:
                self.logger.warning("BehavioralPlanningFacade Didn't reach trigger point yet [%s]. "
                                    "Current localization is [%s]" % (self._trigger_pos, current_pos))

        except Exception as e:
            self.logger.error("BehavioralPlanningFacade error %s" % traceback.format_exc())

