from logging import Logger
from typing import Dict, List

from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.test.constants import TOPIC_PIPELINE_DICTIONARY


class DdsPubSubMock(DdsPubSub):
    def __init__(self, logger: Logger, topic_pipelining=TOPIC_PIPELINE_DICTIONARY):
        # type: (Dict[str, List[str]]) -> None
        """
        Mock for communication layer (DDS)
        :param topic_pipelining: dictionary used to pipeline message from publish topics to their corresponding
        subscribe topics
        """
        self.logger = logger
        self.topic_callback_mapping = {}
        self.topic_msg_mapping = {}
        self.topic_pipeline_mapping = topic_pipelining

    def subscribe(self, topic, callback) -> None:
        """Set a callback on a topic"""
        self.topic_callback_mapping[topic] = callback

    def unsubscribe(self, topic):
        """Unsubscribe (remove a callback) from the given topic"""
        del self.topic_callback_mapping[topic]

    def publish(self, topic, msg):
        """
        Mock passing a message via DDS topics. It actually looks for the destination topic and stores the message
        under its buffer. If callbacks exists, it executes them
        :param topic: Topic to publish message to
        :param msg: the actual message to publish
        """
        destinations = self.topic_pipeline_mapping.get(topic, [topic])

        for destination in destinations:
            callback = self.topic_callback_mapping.get(destination, None)
            if callback is not None:
                callback(msg)
            else:
                self.topic_msg_mapping[destination] = msg

    # this won't support polling and callbacks together
    def get_latest_sample(self, topic, timeout=0):
        return self.topic_msg_mapping[topic]
