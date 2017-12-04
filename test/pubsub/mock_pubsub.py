from collections import defaultdict
from logging import Logger
from typing import Dict, List

from common_data.src.communication.pubsub.pubsub import PubSub


class PubSubMock(PubSub):
    def __init__(self, logger):
        # type: (Logger) -> None
        """
        Mock for communication layer (LCM)
        """
        self.logger = logger
        self.topic_callback_mapping = defaultdict(list)
        self.topic_msg_mapping = {}

    def __del__(self):
        pass

    def subscribe(self, topic, callback, message_type = None, max_data_samples = 10) -> None:
        """Set a callback on a topic"""
        self.topic_callback_mapping[topic].append(callback)

    def unsubscribe(self, topic):
        """Unsubscribe (remove a callback) from the given topic"""
        del self.topic_callback_mapping[topic]

    def publish(self, topic, msg):
        """
        Mock passing a message via LCM topics.
        It actually looks for the topic and stores the message
        under its buffer. If callbacks exists, it executes them
        :param topic: Topic to publish message to
        :param msg: the actual message to publish
        """
        callback_list = self.topic_callback_mapping.get(topic, None)
        for callback in callback_list:
            callback(msg)
        self.topic_msg_mapping[topic] = msg

    # this won't support polling and callbacks together
    def get_latest_sample(self, topic, timeout=0):
        return self.topic_msg_mapping[topic]

    def get_latest_samples_list(self, topic, timeout, max_list_length):
        raise NotImplementedError("Mock does not implement get_latest_samples_list method")