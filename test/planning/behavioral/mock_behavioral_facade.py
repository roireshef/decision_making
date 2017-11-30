from logging import Logger
from typing import Optional

from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.behavioral_facade import BehavioralFacade


class BehavioralFacadeMock(BehavioralFacade):
    """
    Operate according to to policy with an empty dummy behavioral state
    """
    def __init__(self, dds: DdsPubSub, logger: Logger, trajectory_params: TrajectoryParams,
                 visualization_msg: BehavioralVisualizationMsg):
        """
        :param dds: communication layer (DDS) instance
        :param logger: logger
        :param trajectory_params: the trajectory params message to publish periodically
        :param visualization_msg: the visualization message to publish periodically
        """
        super().__init__(dds=dds, logger=logger, policy=None)
        self._trajectory_params = trajectory_params
        self._visualization_msg = visualization_msg

    def _periodic_action_impl(self):
        """
        Publishes the received messages initialized in init
        :return: void
        """
        self._publish_results(self._trajectory_params)
        self._publish_visualization(self._visualization_msg)
