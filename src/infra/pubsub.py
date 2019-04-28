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
            print("!!! " + str(topic))
            topic.register_cb(callback)
        else:
            """
            The event scheduler is named after the callback function and the class that it is contained in. For example, given a class and
            callback function named "Class" and "_callback_function", respectively, the event scheduler will be named
            "Class _callback_function".
            """
            event_scheduler_name = "{}{}".format(callback.__self__.__class__.__name__, callback.__name__)
            print("### " + event_scheduler_name)
            print('Registered EvenScheduler callback: ', event_scheduler_name)
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
