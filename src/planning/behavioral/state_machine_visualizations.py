
from decision_making.src.planning.behavioral.state.driver_initiated_motion_state import DIM_States
from decision_making.src.planning.behavioral.state.lane_change_state import LaneChangeStatus

from decision_making.src.utils.state_machine_visualizer import StateMachineVisualizer
from graphviz import Digraph
from queue import Queue


class DriverInitiatedMotionVisualizer(StateMachineVisualizer):
    def __init__(self, queue: Queue):
        super().__init__(queue=queue, title='DriverInitiatedMotionVisualizer')

    def transform(self, elem: DIM_States) -> Digraph:
        d = Digraph(comment="comment")

        states = {i: s for i, s in enumerate(DIM_States)}

        for i, s in states.items():
            d.node(str(i), str(s), color='Green' if elem == s else 'Grey', style='filled', shape='ellipse')

        for i in range(len(states)-1):
            d.edge(str(i), str(i+1))

        return d


class LaneChangeOnDemandVisualizer(StateMachineVisualizer):
    def __init__(self, queue: Queue):
        super().__init__(queue=queue, title='LaneChangeOnDemandVisualizer')

    def transform(self, elem: LaneChangeStatus) -> Digraph:
        d = Digraph(comment="comment")

        states = {i: s for i, s in enumerate(LaneChangeStatus)}

        for i, s in states.items():
            d.node(str(i), str(s), color='Green' if elem == s else 'Grey', style='filled', shape='ellipse')

        for i in range(len(states)-1):
            d.edge(str(i), str(i+1))

        return d
