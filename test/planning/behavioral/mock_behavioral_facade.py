from logging import Logger

from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.global_constants import BEHAVIORAL_TRAJECTORY_PARAMS_PUBLISH_TOPIC
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.behavioral_facade import BehavioralFacade
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.policy import Policy


class BehavioralFacadeMock(BehavioralFacade):
    """
    Operate according to to policy with an empty dummy behavioral state
    """

    def __init__(self, dds: DdsPubSub, logger: Logger, policy: Policy = None):
        """
        :param policy: decision making component
        :param behavioral_state: initial state of the system. Can be empty, i.e. initialized with default values.
        """
        super().__init__(dds=dds, logger=logger)
        self._policy = policy

    def _periodic_action_impl(self):
        """
        Uses the policy received with an empty behavioral state
        :return: void
        """

        trajectory_params, behavioral_visualization_message = self._policy.plan(
            behavioral_state=BehavioralState())

        self.__publish_results(trajectory_params)
        self.__publish_visualization(behavioral_visualization_message)

    # TODO: make protected and not private
    def __publish_results(self, trajectory_parameters: TrajectoryParams) -> None:
        self.dds.publish(BEHAVIORAL_TRAJECTORY_PARAMS_PUBLISH_TOPIC, trajectory_parameters.serialize())

    def __publish_visualization(self, visualization_message: BehavioralVisualizationMsg) -> None:
    #    self.dds.publish(BEHAVIORAL_VISUALIZATION_TOPIC, visualization_message.serialize())
        pass

