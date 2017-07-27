from src.messages.visualization_message import RvizVisualizationMessage


class RvizVisualizer:
    def __init__(self):
        pass


    # TODO : implement message passing
    def __get_rviz_message(self) -> RvizVisualizationMessage:
        pass

    def __publish_to_rviz(self, results: TrajectoryParameters) -> None:
        pass

    def __publish_visualization(self, visualization_messages: list[RvizVisualizationMessage]) -> None:
        pass