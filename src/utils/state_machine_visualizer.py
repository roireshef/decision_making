import multiprocessing as mp
from abc import abstractmethod
from collections import defaultdict
from io import BytesIO as StringIO
from typing import Any

import cv2
import matplotlib.image as mpimg
import pydotplus
from decision_making.src.utils.multiprocess_visualizer import MultiprocessVisualizer
from graphviz import Digraph


class StateMachineVisualizer(MultiprocessVisualizer):
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


