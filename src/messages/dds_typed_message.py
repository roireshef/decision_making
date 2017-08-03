import inspect
from typing import List, TypeVar
import typing

import numpy as np

from decision_making.src.messages.dds_message import *


class DDSTypedMsg(DDSMsg):
    def serialize(self)->dict:
        """
        used to create the dds message
        :return: dict containing all the fields of the class
        """
        self_dict = self.__dict__
        ser_dict = {}
        for key, val in self_dict.items():
            if isinstance(val, np.ndarray):
                ser_dict[key] = {'array': val.flat.__array__().tolist(), 'shape': val.shape}
            elif inspect.isclass(type(val)) and issubclass(type(val), DDSTypedMsg):
                ser_dict[key] = val.serialize()
            else:
                ser_dict[key] = val
        return ser_dict



    @classmethod
    def deserialize(cls, message: dict):
        """
        used to create an instance of cls represented by the dds message
        :param message: dict containing all fields of the class
        :return: object of type cls, constructed with the arguments from message
        """
        message_copy = message.copy()
        for name, tpe in cls.__init__.__annotations__.items():
            if inspect.isclass(tpe):
                if issubclass(tpe, np.ndarray):
                    message_copy[name] = np.array(message_copy[name]['array']).reshape(message_copy[name]['shape'])
                elif issubclass(tpe, List):
                    message_copy[name] = list(map(lambda d: tpe.__args__[0].deserialize(d), message[name]))
                elif issubclass(tpe, DDSTypedMsg):
                    message_copy[name] = tpe.deserialize(message_copy[name])
        return cls(**message_copy)