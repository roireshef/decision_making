class PubSub:

    @staticmethod
    def subscribe(cls, topic, callback):
        topic.register_cb(callback)

    @staticmethod
    def get_latest_sample(cls, topic, timeout=0):
        return topic.get_latest_sample(timeout*1000)

    @staticmethod
    def publish(cls, topic, data) :
        topic.send(data)

    @staticmethod
    def unsubscribe(cls, topic) :
        topic.unregister_cb(None)
