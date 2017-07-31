import re
from pydoc import locate

import numpy as np

from src.messages.dds_message import *


class DDSNonTypedMsg(DDSMsg):
    def serialize(self):
        """
        used to create the dds message
        :return: dict containing all the fields of the class
        """
        complex_dict = self.__dict__.copy()
        for key, val in complex_dict.items():
            if issubclass(type(val), np.ndarray):
                complex_dict[key] = {'array': val.flat.__array__(), 'shape': val.shape, 'type': 'numpy.ndarray'}
            if issubclass(type(val), DDSMsg):
                item_dict = val.serialize()
                class_type = re.sub('\'>', '', re.sub('<class \'', '', str(type(val))))
                item_dict['type'] = class_type
                complex_dict[key] = item_dict
        return complex_dict

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
                elif issubclass(real_type, DDSMsg):
                    message_copy[key] = real_type.deserialize(val)
        return cls(**message_copy)