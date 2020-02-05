from typing import Callable, Any, Optional, Dict
from rte.python.scheduling.event_scheduler import EventScheduler
import functools
import rte.python.profiler as prof


class PubSub:
    def __init__(self):
        self._event_schedulers: Dict[Any, EventScheduler] = {}

    def subscribe(self, topic, callback: Optional[Callable] = None):
        """
        Subscribes the given callback to the given pubsub topic. Once this topic will be published, it will be received
        and processed by the callback method.
        :param topic:
        :param callback:
        :return:
        """

        if callback is None:
            topic.register_cb(callback)
        else:
            """
            The event scheduler is named after the callback function and the class that it is contained in. For example, given a class and
            callback function named "Class" and "_callback_function", respectively, the event scheduler will be named
            "Class _callback_function".
            """
            event_scheduler_name = "{}{}".format(callback.__self__.__class__.__name__, callback.__name__)
            event_scheduler = EventScheduler(event_scheduler_name)

            if prof.is_enabled(prof.Category.Callback):
                callback = self._wrap_cb_with_profiling(callback, topic)

            event_scheduler.register_cb(topic, callback)

            self._event_schedulers[topic] = event_scheduler

    @staticmethod
    def get_latest_sample(topic, timeout: float = 0):
        """
        Access the data structure holding the given topic messages and pull the latest sample from this data structure.
        :param topic:
        :param timeout:
        :return:
        """
        data = topic.get_latest_sample(timeout*1000)
        with prof.time_range("[message] " + str(topic.msg_type.__name__), category=prof.Category.Communication):
            pass
        return data

    @staticmethod
    @prof.ProfileFunction()
    def publish(topic, data: Any):
        """
        Publish the given data object in the given topic
        :param topic:
        :param data:
        :return:
        """
        topic.send(data)

    def unsubscribe(self, topic):
        """
        Unsuscribes ALL(!) callbacks from the topic given as argument to this method.
        :param topic:
        :return:
        """
        # TODO Implement unsubscribe from a specific callback if required
        if topic not in self._event_schedulers.keys():
            topic.unregister_cb(None)
        else:
            self._event_schedulers[topic].unregister_cb()

    @staticmethod
    def _wrap_cb_with_profiling(callback, name):
        _prof_mark_cb = prof.TimeRange("[event] %s", name, category=prof.Category.Callback)

        @functools.wraps(callback)
        def cb_wrapper(*args, **kwargs):
            with _prof_mark_cb:
                return callback(*args, **kwargs)
        return cb_wrapper
