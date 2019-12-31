import multiprocessing as mp
import time
from abc import abstractmethod

from pandas import DataFrame
from typing import Any


class AsyncVisualizer(mp.Process):
    def __init__(self, queue_len: int = 20):
        """
        Interface for a visualizer that uses a dedicated process for visualization. It opens a multiprocess Queue used
        to communicate the data for visualization and a Lock used to (optionally) lock client process waiting for the
        visualization figure to load
        :param queue_len:
        """
        super().__init__()

        self._queue = mp.Queue(queue_len)
        self._is_running = mp.Value('b', False)
        self._figure_lock = mp.Lock()

        self._data = self._init_data()

    @property
    def queue(self):
        return self._queue

    def run(self):
        # Here we lock <self._figure_lock> while we initialize the figure and release it after the figure
        # has been initialized. The client process can choose to call async_visualizer.wait_for_figure_lock() if it
        # wants to wait for figure-initialization
        self._figure_lock.acquire()
        self._init_fig()
        self._refresh_fig()
        self._figure_lock.release()

        self._is_running.value = True
        while self._is_running.value:
            # if queue is empty, refresh figure to avoid greying it out and sleep for some short period
            if self._queue.empty():
                time.sleep(0.01)
                self._refresh_fig()
                continue

            # read from queue into the buffer <self._data> until it is empty
            while self._is_running.value and not self._queue.empty():
                elem = self._queue.get_nowait()
                self._update_data(elem)

            # use <self._data> to visualize and refresh the figure
            self._update_fig()
            self._refresh_fig()

        self._destroy_fig()
        self._queue.close()
        self.kill()

    def stop(self):
        self._is_running.value = False

    def append(self, elem):
        """
        adds another data-point to the queue that will be used for updating the plots
        :param elem:
        :return:
        """
        self._queue.put(elem)

    def wait_for_figure_lock(self):
        """
        utility method used to lock the user's process while plots are being initialized. When lock is being acquired,
        if it's already acquired by other process, then interpreter waits for it to be released first (and therefor
        waits at the lock.acquire() line.
        """
        self._figure_lock.acquire()
        self._figure_lock.release()

    # USER CUSTOMIZABLE METHODS

    def _init_data(self) -> Any:
        """
        initializes the local data buffer used for visualization from new elements in the queue. can be of any type
        :return:
        """
        return DataFrame()

    def _update_data(self, elem: Any):
        """
        adds an element to the local buffer. if _init_data returns any class instance other than List or DataFrame, this
        method should be overridden accordingly
        :param elem:
        :return:
        """
        self._data = self._data.append(elem)

    @abstractmethod
    def _init_fig(self):
        """
        method for user to specify initialization for visualization figure
        """
        pass

    @abstractmethod
    def _update_fig(self):
        """
        method for user to specify how to update visualization figure based on current self.data contents
        """
        pass

    @abstractmethod
    def _refresh_fig(self):
        """
        method for user to specify initialization how to refresh visualization figure
        """
        pass

    @abstractmethod
    def _destroy_fig(self):
        """
        method for user to specify initialization how to close and destroy visualization figure
        """
        pass


class DummyVisualizer(AsyncVisualizer):
    def __init__(self):
        """
        A dummy visualizer the exposes the same interface of MultiprocessVisualizer, but does nothing.
        This is useful for a "null instantiation" of a visualizer, to not impact code.
        """
        pass

    @property
    def queue(self):
        return None

    def start(self) -> None:
        pass

    def run(self):
        pass

    def stop(self):
        pass

    def append(self, elem):
        pass

    def wait_for_figure_lock(self):
        pass

    def _init_fig(self):
        pass

    def _update_fig(self):
        pass

    def _refresh_fig(self):
        pass

    def _destroy_fig(self):
        pass
