from decision_making.src.state.state import State
from common_data.lcm.generatedFiles.gm_lcm import LcmStateVisualizationMsg


class StateVisualizationMsg:
    def __init__(self, state):
        # type: (State) -> None
        """
        The struct used for communicating the state to the visualizer. Should be published by the state module.
        TODO - this is a temporary solution for supporting the python 2.7 requirement from rviz.
        :param state: of type State.
        """
        self.state = state

    def to_lcm(self) -> LcmStateVisualizationMsg:
        lcm_msg = LcmStateVisualizationMsg()

        lcm_msg.state = self.state.to_lcm()

        return lcm_msg

    @classmethod
    def from_lcm(cls, lcmMsg: LcmStateVisualizationMsg):
        return cls(State.from_lcm(lcmMsg.state))

