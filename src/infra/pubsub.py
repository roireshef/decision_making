class PubSub:

    def __init__(self, message_types):
        self._message_types = message_types

    def subscribe(self, topic, callback, message_type= None, max_data_samples= 10):
        if message_type is None:
            message_type = self._message_types[topic]
        message_type.register_cb(callback)

    def _get_latest_sample(self, topic, timeout=0):
        return topic.get_latest_sample(timeout*1000)

    def publish(self, topic, data) :
        message_type = self._message_types[topic]
        message_type.send(data)

    def get_latest_samples_list(self, topic, timeout, max_list_length) :
        pass

    def unsubscribe(self, topic) :
        message_type = self._message_types[topic]
        message_type.unregister_cb(None)