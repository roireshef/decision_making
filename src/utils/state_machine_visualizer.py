import multiprocessing as mp
from abc import abstractmethod

from graphviz import Digraph
import pydotplus

from io import BytesIO as StringIO
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from typing import Any, Tuple


class StateMachineVisualizer(mp.Process):
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
        self.plot_num = plot_num

    def run(self):
        self.is_running.value = True

        while self.is_running.value:
            elem = self.queue.get()
            self._view(self.transform(elem))

    def stop(self):
        self.is_running.value = False
        for fig in self.fig:
            plt.close(fig)
        self.kill()

    @abstractmethod
    def transform(self, elem: Any) -> Tuple[int, Digraph]:
        """
        abstract class that user implements for custom transofmation from any object to Digraph for visualization
        :param elem: any obj that will be stored in <self.queue>
        :return: a Digraph visualization object
        """
        pass

    def _view(self, sub_plot_graph: Tuple[int, Digraph]):

        sub_plot_num = sub_plot_graph[0]-1
        graph = sub_plot_graph[1]
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

        # initialize window used to plot images
        if self.im[sub_plot_num] is None:
            self.fig[sub_plot_num], ax = plt.subplots(1,1)
            self.im[sub_plot_num] = ax.imshow(img)
        else:
            self.im[sub_plot_num].set_data(img)
        self.fig[sub_plot_num].canvas.draw_idle()
        plt.pause(0.1)

