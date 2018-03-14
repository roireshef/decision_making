import os
import json
import time
import threading

import lcm

from decision_making.test.lcm_trajectory_player.gm_lcm import LcmNumpyArray
LCM_CONFIG_FILES_PATH = os.path.dirname(os.path.realpath(__file__)) + '/config/'

class LcmPubSub():

    # class SubUnsubRequest:
    #     def __init__(self, is_sub, topic, callback = None, message_type = None, max_data_samples = None):
    #         self.is_sub = is_sub
    #         self.topic = topic
    #         self.callback = callback
    #         self.message_type = message_type
    #         self.max_data_samples = max_data_samples

    # class LcmSubscription:
    #     def __init__(self, callback, lcm_type, max_data_samples, subscription):
    #         self.callbacks = set()
    #         self.add_callback(callback)
    #         self.lcm_type = lcm_type
    #         self.data_samples = []
    #         self.max_data_samples = max_data_samples
    #         self.subscription = subscription
    #
    #     def add_callback(self, callback):
    #         self.callbacks.add(callback)

    def __init__(self, json_config_file, domain_id=0):
        with open(os.path.join(LCM_CONFIG_FILES_PATH, json_config_file)) as data_file:
            config_data = json.load(data_file)

        split_socket = config_data["socket"].rsplit(":", 1)
        split_socket[-1] = str(int(split_socket[-1]) + int(domain_id))
        self.lcm = lcm.LCM(":".join(split_socket))
        self.thread = None
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.subscriptions = {}
        self.subscribe_requests = []

    def __del__(self):
        for topic, subscription in self.subscriptions.items():
            self.lcm.unsubscribe(subscription.subscription)

    # def subscribe(self, topic, callback = None, max_data_samples = 10):
    #     try:
    #         message_type = MessageTypes[topic]
    #     except KeyError:
    #         raise ValueError("Topic {} is not defined in LcmPubSub".format(topic))
    #
    #     if max_data_samples <= 0:
    #         raise ValueError("Invalid max_data_samples=%d" % max_data_samples)
    #
    #     sub_request = self.SubUnsubRequest(True, topic, callback, message_type, max_data_samples)
    #
    #     self.lock.acquire()
    #
    #     # We should start the background thread if it was not started.
    #     if self.thread is None:
    #         self.thread = threading.Thread(target = self.__thread_loop)
    #         self.thread.start()
    #
    #     self.subscribe_requests.append(sub_request)
    #
    #     self.lock.release()

    # def unsubscribe(self, topic):
    #     # TODO: add unsubscribe of a specific callback
    #
    #     unsub_request = self.SubUnsubRequest(False, topic)
    #
    #     self.lock.acquire()
    #
    #     self.subscribe_requests.append(unsub_request)
    #
    #     self.lock.release()

    def publish(self, topic, data):
        self.lcm.publish(topic, data.encode())

    # def get_latest_sample(self, topic, timeout = 0):
    #     endTime = time.time() + timeout
    #     remained_timeout = timeout
    #
    #     self.lock.acquire()
    #
    #     sample = None
    #     while True:
    #         if topic not in self.subscriptions:
    #             break
    #
    #         if len(self.subscriptions[topic].data_samples) > 0:
    #             sample = self.subscriptions[topic].data_samples[-1]
    #             break
    #
    #         if remained_timeout <= 0:
    #             break
    #
    #         self.condition.wait(remained_timeout)
    #         remained_timeout = endTime - time.time()
    #
    #     self.lock.release()
    #
    #     return sample
    #
    # def get_latest_samples_list(self, topic, timeout, max_list_length):
    #     endTime = time.time() + timeout
    #     remained_timeout = timeout
    #
    #     self.lock.acquire()
    #
    #     sample_list = []
    #     while True:
    #         if topic not in self.subscriptions:
    #             break
    #
    #         if len(self.subscriptions[topic].data_samples) > 0:
    #             sample_list = self.subscriptions[topic].data_samples[-max_list_length:]
    #             break
    #
    #         if remained_timeout <= 0:
    #             break
    #
    #         self.condition.wait(remained_timeout)
    #         remained_timeout = endTime - time.time()
    #
    #     self.lock.release()
    #
    #     return sample_list
    #
    # def __thread_loop(self):
    #     while True:
    #         self.lcm.handle_timeout(100) #100 ms
    #         self.__processSubscribeRequests()
    #
    # def __processSubscribeRequests(self):
    #     self.lock.acquire()
    #
    #     for sub_unsub_req in self.subscribe_requests:
    #         if sub_unsub_req.is_sub == True:
    #             if sub_unsub_req.topic not in self.subscriptions:
    #                 self.subscriptions[sub_unsub_req.topic] = self.LcmSubscription(
    #                     callback = sub_unsub_req.callback,
    #                     lcm_type = sub_unsub_req.message_type,
    #                     max_data_samples = sub_unsub_req.max_data_samples,
    #                     subscription = self.lcm.subscribe(sub_unsub_req.topic, self.__general_callback))
    #             else:
    #                 self.subscriptions[sub_unsub_req.topic].add_callback(sub_unsub_req.callback)
    #         else:
    #             if sub_unsub_req.topic in self.subscriptions:
    #                 self.lcm.unsubscribe(self.subscriptions[sub_unsub_req.topic].subscription)
    #                 self.subscriptions.pop(sub_unsub_req.topic)
    #
    #     # self.subscribe_requests.clear()
    #     del self.subscribe_requests[:]
    #     self.lock.release()
    #
    # def __general_callback(self, topic, data):
    #     callbacks = None
    #
    #     self.lock.acquire()
    #
    #     decoded_data = None
    #     if topic in self.subscriptions:
    #         subscription = self.subscriptions[topic]
    #         decoded_data = subscription.lcm_type.decode(data)
    #
    #         subscription.data_samples.append(decoded_data)
    #         while len(subscription.data_samples) > subscription.max_data_samples:
    #             subscription.data_samples.pop(0)
    #
    #         callbacks = subscription.callbacks.copy()
    #         self.condition.notifyAll()
    #
    #     self.lock.release()
    #
    #     if callbacks is not None:
    #         for callback in callbacks:
    #             if callback is not None:
    #                 callback(decoded_data)
