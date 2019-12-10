import multiprocessing as mp

from decision_making.src.planning.behavioral.state.driver_initiated_motion_state import DIM_States
from decision_making.src.planning.behavioral.state.lane_change_state import LaneChangeStatus

from decision_making.src.utils.state_machine_visualizer import StateMachineVisualizer
from graphviz import Digraph
from typing import Tuple, Union


class MultiVisualizer(StateMachineVisualizer):
    def __init__(self, queue: mp.SimpleQueue):
        super().__init__(queue=queue, title='MultiVisualizer', plot_num=2)

    def transform(self, elem: Union[LaneChangeStatus, DIM_States]) -> Tuple[int, Digraph]:
        d = Digraph(comment="comment")

        states = {}
        sub_plot_num = 0
        if isinstance(elem, LaneChangeStatus):
            states = {i: s for i, s in enumerate(LaneChangeStatus)}
            sub_plot_num = 1

        elif isinstance(elem, DIM_States):
            states = {i: s for i, s in enumerate(DIM_States)}
            sub_plot_num = 2

        for i, s in states.items():
            d.node(str(i), str(s).split('.')[-1], color='Green' if elem == s else 'Grey', style='filled', shape='ellipse')

        for i in range(len(states)-1):
            d.edge(str(i), str(i+1))

        d.edge(str(len(states)-1), str(0))

        return sub_plot_num, d
