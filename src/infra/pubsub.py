class PubSub:

    @classmethod
    def subscribe(cls, topic, callback):
        topic.register_cb(callback)

    @classmethod
    def get_latest_sample(cls, topic, timeout=0):
        return topic.get_latest_sample(timeout*1000)

    @classmethod
    def publish(cls, topic, data) :
        topic.send(data)

    @classmethod
    def get_latest_samples_list(cls, topic, timeout, max_list_length) :
        pass

    @classmethod
    def unsubscribe(cls, topic) :
        topic.unregister_cb(None)
