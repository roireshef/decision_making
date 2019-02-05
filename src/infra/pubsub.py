class PubSub:

    @staticmethod
    def subscribe(topic, callback):
        topic.register_cb(callback)

    @staticmethod
    def get_latest_sample(topic, timeout=0):
        return topic.get_latest_sample(timeout*1000)

    @staticmethod
    def publish(topic, data) :
        topic.send(data)

    @staticmethod
    def unsubscribe(topic) :
        topic.unregister_cb(None)
