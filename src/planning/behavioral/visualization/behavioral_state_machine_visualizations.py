from typing import Union

import matplotlib.pyplot as plt
from decision_making.src.planning.behavioral.state.driver_initiated_motion_state import DIM_States
from decision_making.src.planning.behavioral.state.lane_change_state import LaneChangeStatus
from decision_making.src.utils.state_machine_visualizer import StateMachineVisualizer
from graphviz import Digraph


class DIMAndLCoDVisualizer(StateMachineVisualizer):
    def __init__(self, plot_num: int = 2):
        """
        Visualizer that visualizes DIM and LCoD state machine graphs
        """
        super().__init__(plot_num)

    def _init_fig(self):
        self.fig = plt.figure(figsize=(10, 8), num="Planning Agent State Machine Status")
        self.fig.suptitle('Agent\'s State-Machine Status', fontsize=16)

        lcod_ax = plt.subplot(1, 2, 1)
        lcod_ax.set_title('Lane Change on Demand')
        lcod_ax.axis('off')

        dim_ax = plt.subplot(3, 2, 4)
        dim_ax.set_title('Driver-initiated Motion')
        dim_ax.axis('off')

        self.fig.tight_layout()
        self.ax = [lcod_ax, dim_ax]
        self.fig.show()

    @staticmethod
    def _transform(elem: Union[LaneChangeStatus, DIM_States]) -> Digraph:
        d = Digraph(comment="comment")

        states = {}
        if isinstance(elem, LaneChangeStatus):
            states = {i: s for i, s in enumerate(LaneChangeStatus)}

        elif isinstance(elem, DIM_States):
            states = {i: s for i, s in enumerate(DIM_States)}

        for i, s in states.items():
            d.node(str(i), str(s).split('.')[-1], color='Green' if elem == s else 'Grey', style='filled',
                   shape='ellipse')

        for i in range(len(states)-1):
            d.edge(str(i), str(i+1))

        d.edge(str(len(states)-1), str(0), _attributes={"constraint": "false"})

        if isinstance(elem, LaneChangeStatus):
            for i in [1, 2, 3]:
                d.edge(str(i), str(0), "(abort)", _attributes={"constraint": "false"})

        return d

    @staticmethod
    def _type_to_index(elem: Union[LaneChangeStatus, DIM_States]) -> int:
        if isinstance(elem, LaneChangeStatus):
            return 0
        elif isinstance(elem, DIM_States):
            return 1
        else:
            return -1

