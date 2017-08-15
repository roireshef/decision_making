from decision_making.src.messages.dds_nontyped_message import DDSNonTypedMsg


class StateVisualizationMsg(DDSNonTypedMsg):
    def __init__(self, state):
        """
        The struct used for communicating the state to the visualizer. Should be published by the state module.
        TODO - this is a temporary solution for supporting the python 2.7 requirement from rviz.
        :param state: of type State.
        """
        self.state = state


