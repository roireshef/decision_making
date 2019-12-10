import time
from abc import abstractmethod

from graphviz import Digraph
import pydotplus

from io import BytesIO as StringIO
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from typing import Any
from queue import Queue


class StateMachineVisualizer():
    def __init__(self, queue: Queue, title: str):
        """
        A new process that opens a new visualization window and plots graphviz plots inside
        :param max_queue_len: max elements in the queue that is used for communicating with this visualizer
        :param title: The visualization window's title
        """
        super().__init__()
        self.title = title
        self.im = None
        self.fig = None

    def update(self, elem):
        graph = self.transform(elem)
        self._view(graph)

    @abstractmethod
    def transform(self, elem: Any) -> Digraph:
        """
        abstract class that user implements for custom transofmation from any object to Digraph for visualization
        :param elem: any obj that will be stored in <self.queue>
        :return: a Digraph visualization object
        """
        pass

    def _view(self, graph: Digraph):

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
        if self.im is None:
            self.fig, ax = plt.subplots(1,1)
            self.im = ax.imshow(img)
        else:
            self.im.set_data(img)
            self.fig.canvas.draw_idle()
            plt.pause(0.001)


