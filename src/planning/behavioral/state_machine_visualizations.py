from typing import Any

from decision_making.src.utils.state_machine_visualizer import StateMachineVisualizer
from graphviz import Digraph


class DriverInitiatedMotionVisualizer(StateMachineVisualizer):
    def transform(self, elem: Any) -> Digraph:
        pass
