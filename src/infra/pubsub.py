from typing import Callable, Any, List, Optional
from rte.python.scheduling.event_scheduler import EventScheduler


class PubSub:
    def __init__(self):
        self._event_schedulers: List[EventScheduler] = []

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
            event_scheduler_name = "{} {}".format(callback.__self__.__class__.__name__, callback.__name__)
            event_scheduler = EventScheduler(event_scheduler_name)
            event_scheduler.register_cb(topic, callback)

            self._event_schedulers.append(event_scheduler)

    @staticmethod
    def get_latest_sample(topic, timeout: float = 0):
        """
        Access the data structure holding the given topic messages and pull the latest sample from this data structure.
        :param topic:
        :param timeout:
        :return:
        """
        return topic.get_latest_sample(timeout*1000)

    @staticmethod
    def publish(topic, data: Any):
        """
        Publish the given data object in the given topic
        :param topic:
        :param data:
        :return:
        """
        topic.send(data)

    @staticmethod
    def unsubscribe(topic):
        """
        Unsuscribes ALL(!) callbacks from the topic given as argument to this method.
        :param topic:
        :return:
        """
        # TODO Implement unsubscribe from a specific callback if required
        topic.unregister_cb(None)
