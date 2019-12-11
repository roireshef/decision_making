import multiprocessing as mp
from abc import abstractmethod

from graphviz import Digraph
import pydotplus

from io import BytesIO as StringIO
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from typing import Any


class StateMachineVisualizer(mp.Process):
    TICK_TO_PLOT = 10

    def __init__(self, queue: mp.SimpleQueue, title: str, plot_num: int = 1):
        """
        A new process that opens a new visualization window and plots graphviz plots inside
        :param max_queue_len: max elements in the queue that is used for communicating with this visualizer
        :param title: The visualization window's title
        """
        super().__init__()
        self.title = title
        self.queue = queue
        self.is_running = mp.Value('b', False)
        self.im = [None] * plot_num
        self.fig = [None] * plot_num
        self.previous_elem = [None] * plot_num
        self.previous_img = [None] * plot_num
        self.tick_from_previous_img = [0] * plot_num
        self.plot_num = plot_num

    def run(self):
        self.is_running.value = True

        while self.is_running.value:
            try:
                elem = self.queue.get()
                type_index = self.type_to_index(elem)
                if type_index >= 0:
                    keep_previous_fig = (self.previous_elem[type_index] is not None) and (elem.value == self.previous_elem[type_index].value)
                    self._view(self.transform(elem) if not keep_previous_fig else None, type_index)
                self.previous_elem[type_index] = elem
            except:
                pass

    def stop(self):
        self.is_running.value = False
        for fig in self.fig:
            plt.close(fig)
        self.kill()

    @abstractmethod
    def type_to_index(self, elem: Any) -> int:
        """
        abstract class that user implements to return the index of the element's type. Used to compare to previous element
        :param elem: any obj that will be stored in <self.queue>
        :return: type index
        """
        pass

    @abstractmethod
    def transform(self, elem: Any) -> Digraph:
        """
        abstract class that user implements for custom transofmation from any object to Digraph for visualization
        :param elem: any obj that will be stored in <self.queue>
        :return: a Digraph visualization object
        """
        pass

    def _view(self, graph: Digraph, sub_plot_num: int):
        """
        draw the relevant graph
        :param graph: the graph to draw, or null if previous graph is retained
        :param sub_plot_num: window number to draw into
        :return: Nothing
        """
        if graph is not None:
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

            if self.im[sub_plot_num] is None:
                # initialize window used to plot images
                self.fig[sub_plot_num], ax = plt.subplots(1, 1)
                self.im[sub_plot_num] = ax.imshow(img)
            else:
                self.im[sub_plot_num].set_data(img)
            self.previous_img[sub_plot_num] = img
            self.tick_from_previous_img[sub_plot_num] = 0
        else:
            # if graph is None then use previous figure
            if self.tick_from_previous_img[sub_plot_num] > self.TICK_TO_PLOT:
                self.tick_from_previous_img[sub_plot_num] = 0
                self.im[sub_plot_num].set_data(self.previous_img[sub_plot_num])
            else:
                self.tick_from_previous_img[sub_plot_num] += 1
                return
        self.fig[sub_plot_num].canvas.draw_idle()
        plt.pause(0.1)
