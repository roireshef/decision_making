from typing import Dict, List

from common_data.dds.python.Communication.ddspubsub import DdsPubSub


class DdsPubSubMock(DdsPubSub):

    def __init__(self, topic_pipelining):
        # type: (Dict[str, List[str]]) -> None
        self.topic_callback_mapping = {}
        self.topic_msg_mapping = {}
        self.topic_pipeline_mapping = topic_pipelining

    def send_message(self, topic, msg):
        destinations = self.topic_pipeline_mapping.get(topic, [topic])

        for destination in destinations:
            callback = self.topic_callback_mapping.get(destination, None)
            if callback is not None:
                callback(msg)
            else:
                self.topic_msg_mapping[destination] = msg

    def subscribe(self, topic, callback):
        self.topic_callback_mapping[topic] = callback

    def unsubscribe(self, topic):
        """Unsubscribe from the given topic"""
        del self.topic_callback_mapping[topic]

    def publish(self, topic, data):
        self.send_message(topic, data)

    # this won't support polling and callbacks together
    def get_latest_sample(self, topic, timeout=0):
        return self.topic_msg_mapping[topic]
