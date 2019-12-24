from typing import Any
from typing import Union

import matplotlib.pyplot as plt
from decision_making.src.planning.behavioral.state.driver_initiated_motion_state import DIM_States
from decision_making.src.planning.behavioral.state.lane_change_state import LaneChangeStatus
from decision_making.src.utils.state_machine_visualizer import StateMachineVisualizer
from graphviz import Digraph


class BehavioralStateMachineVisualizer(StateMachineVisualizer):
    def __init__(self, plot_num: int = 2):
        """
        Visualizer that visualizes DIM and LCoD state machine graphs
        """
        self.plot_num = plot_num

        super().__init__()

        self.im = [None] * self.plot_num
        self.ax = [None] * self.plot_num

    def _init_data(self) -> Any:
        return [None] * self.plot_num

    def _update_data(self, elem: Any):
        self._data[self._type_to_index(elem)] = self._render_digraph(self._transform(elem))

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

    def _update_fig(self):
        for idx, img in enumerate(self._data):
            if img is None:
                continue

            if self.im[idx] is None:
                # initialize window used to plot images
                self.im[idx] = self.ax[idx].imshow(img)
            else:
                self.im[idx].set_data(img)

    def _refresh_fig(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _destroy_fig(self):
        plt.close(self.fig)

    def _transform(self, elem: Union[LaneChangeStatus, DIM_States]) -> Digraph:
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

