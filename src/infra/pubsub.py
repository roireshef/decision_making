from typing import Callable, Any


class PubSub:

    @staticmethod
    def subscribe(topic, callback: Callable):
        """
        Subscribes the given callback to the given pubsub topic. Once this topic will be published, it will be received
        and processed by the callback method.
        :param topic:
        :param callback:
        :return:
        """
        topic.register_cb(callback)

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
