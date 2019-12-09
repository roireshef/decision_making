import multiprocessing as mp

from decision_making.src.planning.behavioral.state.driver_initiated_motion_state import DIMStates
from decision_making.src.utils.state_machine_visualizer import StateMachineVisualizer
from graphviz import Digraph


class DriverInitiatedMotionVisualizer(StateMachineVisualizer):
    def __init__(self, queue: mp.Queue):
        super().__init__(queue=queue, title='DriverInitiatedMotionVisualizer')

    def transform(self, elem: DIMStates) -> Digraph:
        d = Digraph(comment="comment")

        states = {i: s for i, s in enumerate(DIMStates)}

        for i, s in states.items():
            d.node(str(i), str(s), color='Green' if elem == s else 'Grey', style='filled', shape='ellipse')

        for i in range(len(states)-2):
            d.edge(str(i), str(i+1))

        return d