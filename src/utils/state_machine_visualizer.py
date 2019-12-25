from abc import abstractmethod
from io import BytesIO as StringIO
from typing import Any

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pydotplus
from decision_making.src.utils.multiprocess_visualizer import MultiprocessVisualizer
from graphviz import Digraph


class StateMachineVisualizer(MultiprocessVisualizer):
    def __init__(self, plot_num):
        """
        Visualizer that visualizes a single Matplotlib figure with flexible number of subplots (axes)
        :param plot_num: number of different subplots in the figure (visualization window), each one can hold a
        rendering of a Digraph object
        """
        self.plot_num = plot_num

        super().__init__()

        self.fig = None                                     # Matplotlib figure (window)
        self.im = [None] * self.plot_num                    # List of images (imshow handles)
        self.ax = [None] * self.plot_num                    # List of Matplotlib Axes
        self.last_drawn_elem = [None] * self.plot_num       # Buffer of last drawn element of each type

    @staticmethod
    @abstractmethod
    def _type_to_index(elem: Any) -> int:
        """
        Mapping function for an element pulled from the queue should with the index of the correct axis in <self.ax>.
        # Should be implemented by user #
        :param elem: an element from <self._queue>
        :return: index of the correct axis
        """
        pass

    @staticmethod
    @abstractmethod
    def _transform(elem: Any) -> Digraph:
        """
        Mapping of an element pulled from the queue to a Digraph object
        # Should be implemented by user #
        :param elem: an element from <self._queue>
        :return: Digraph (graphviz) object
        """
        pass

    @abstractmethod
    def _init_fig(self):
        """
        Method to initialize axes (plots), customizable for user provided layout.
        # Should be implemented by user #, and should include:
        * Initialization of figure and storing it in <self.fig>. example:
            self.fig = plt.figure()
        * Initialization of figure-axes, and storing them in <self.ax>. example:
            lcod_ax = plt.subplot(1, 2, 1)
            ...
            dim_ax = plt.subplot(3, 2, 4)
            ...
            self.ax = [lcod_ax, dim_ax]

            order matters here - and should comply with self._type_to_index where the type of element in the queue
            should be mapped into the index of the correct axis in self.ax
        * This method should end with: self.fig.show()
        """
        pass

    def _init_data(self) -> Any:
        return [None] * self.plot_num

    def _update_data(self, elem: Any):
        self._data[self._type_to_index(elem)] = elem

    def _update_fig(self):
        for idx, elem in enumerate(self._data):
            # avoid redrawing the same element (if it hasn't changed)
            if elem is None or self._data[idx] == self.last_drawn_elem[idx]:
                continue

            # draw element - if first time - plot, otherwise - update plot
            img = self._render_digraph(self._transform(elem))
            if self.im[idx] is None:
                # initialize window used to plot images
                self.im[idx] = self.ax[idx].imshow(img)
            else:
                self.im[idx].set_data(img)

            self.last_drawn_elem[idx] = elem

    def _refresh_fig(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _destroy_fig(self):
        plt.close(self.fig)

    def _render_digraph(self, graph: Digraph):
        """
        transform a Digraph object into an image plottable by cv2 or matplotlib
        :param graph: the graph to draw, or null if previous graph is retained
        :param sub_plot_num: window number to draw into
        :return: pixels matrix
        """
        # convert from networkx -> pydot
        dotgraph = pydotplus.graph_from_dot_data(graph.source)
        png_str = dotgraph.create_png(prog='dot')

        # render pydot by calling dot, no file saved to disk
        sio = StringIO()
        sio.write(png_str)
        sio.seek(0)
        img = mpimg.imread(sio)

        # Manipulate image
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        ratio = img.shape[0] / float(img.shape[1])
        new_width = 1080
        new_height = int(new_width * ratio)
        img = cv2.resize(img, (new_width, new_height))

        return img


