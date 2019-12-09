import multiprocessing as mp
import time
from abc import abstractmethod

from graphviz import Digraph
import pydotplus

from io import BytesIO as StringIO
import cv2
import matplotlib.image as mpimg
from typing import Any


class StateMachineVisualizer(mp.Process):
    def __init__(self, title: str, max_queue_len: int = 10):
        """
        A new process that opens a new visualization window and plots graphviz plots inside
        :param max_queue_len: max elements in the queue that is used for communicating with this visualizer
        :param title: The visualization window's title
        """
        super().__init__()
        self.title = title
        self.queue = mp.Queue(max_queue_len)
        self.is_running = mp.Value('b', False)

    def run(self):
        self.is_running.value = True
        elem = None

        while self.is_running.value:

            # TODO: This is hacky, should change to synchronized queue
            while not self.queue.empty():
                elem = self.queue.get()

            if elem is None:
                time.sleep(0.01)
                continue

            self._view(self.transform(elem))

    def stop(self):
        self.is_running.value = False

    @abstractmethod
    def transform(self, elem: Any) -> Digraph:
        """
        abstract class that user implements for custom transofmation from any object to Digraph for visualization
        :param elem: any obj that will be stored in <self.queue>
        :return: a Digraph visualization object
        """
        pass

    def _view(self, graph: Digraph):
        image_obj = None

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

        if image_obj is None:
            cv2.imshow("Tree", img)
            cv2.waitKey(1)
