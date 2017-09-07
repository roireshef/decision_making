from decision_making.src.messages.dds_typed_message import DDSTypedMsg
from decision_making.src.state.state import State


class StateVisualizationMsg(DDSTypedMsg):
    def __init__(self, state):
        # type: (State) -> None
        """
        The struct used for communicating the state to the visualizer. Should be published by the state module.
        TODO - this is a temporary solution for supporting the python 2.7 requirement from rviz.
        :param state: of type State.
        """
        self.state = state


