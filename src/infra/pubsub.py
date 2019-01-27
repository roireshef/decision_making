class PubSub:

    #def __init__(self, message_types):
        #self._message_types = message_types

    @classmethod
    def subscribe(cls, topic, callback):
        #if message_type is None:
        #    message_type = self._message_types[topic]
        topic.register_cb(callback)

    @classmethod
    def get_latest_sample(cls, topic, timeout=0):
        return topic.get_latest_sample(timeout*1000)

    @classmethod
    def publish(cls, topic, data) :
        #message_type = self._message_types[topic]
        print("222222222222222222222222222")
        topic.send(data)

    @classmethod
    def get_latest_samples_list(cls, topic, timeout, max_list_length) :
        pass

    @classmethod
    def unsubscribe(cls, topic) :
        #message_type = self._message_types[topic]
        topic.unregister_cb(None)
