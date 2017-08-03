import re
from pydoc import locate

import numpy as np

from decision_making.src.messages.dds_message import *


class DDSNonTypedMsg(DDSMsg):
    def serialize(self):
        """
        used to create the dds message
        :return: dict containing all the fields of the class
        """
        ser_dict = {}
        for key, val in self.__dict__.items():
            if issubclass(type(val), np.ndarray):
                ser_dict[key] = {'array': val.flat.__array__().tolist, 'shape': val.shape, 'type': 'numpy.ndarray'}
            elif issubclass(type(val), list):
                ser_dict[key] = {'iterable': list(map(lambda x: x.serialize(), val)), 'type': type(val).__name__}
            elif issubclass(type(val), DDSMsg):
                ser_dict[key] = val.serialize()
            else:
                ser_dict[key] = val
        ser_dict['type'] = self.__module__ + "." + self.__class__.__name__
        return ser_dict

    @classmethod
    def deserialize(cls, message):
        """
        used to create an instance of cls represented by the dds message
        :param message: dict containing all fields of the class
        :return: object of type cls, constructed with the arguments from message
        """
        message_copy = message.copy()
        message_copy.pop('type', None)
        for key, val in message_copy.items():
            if isinstance(val, dict):   # instance was created from a class
                real_type = locate(val['type'])
                if issubclass(real_type, np.ndarray):
                    message_copy[key] = np.array(val['array']).reshape(val['shape'])
                elif issubclass(real_type, list):
                    message_copy[key] = list(map(lambda d: locate(d['type']).deserialize(d), val['iterable']))
                elif issubclass(real_type, DDSMsg):
                    message_copy[key] = real_type.deserialize(val)
        return cls(**message_copy)