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
        for name, tpe in self.__init__.__annotations__.items():
            if inspect.isclass(tpe):
                if issubclass(tpe, np.ndarray):
                    ser_dict[name] = {'array': self_dict[name].flat.__array__().tolist(),
                                      'shape': list(self_dict[name].shape)}
                elif issubclass(tpe, list):
                    ser_dict[name] = list(map(lambda x: x.serialize(), self_dict[name]))
                elif inspect.isclass(tpe) and issubclass(tpe, DDSTypedMsg):
                    ser_dict[name] = self_dict[name].serialize()
                else:
                    ser_dict[name] = self_dict[name]
        return ser_dict



    @classmethod
    def deserialize(cls, message: dict):
        """
        used to create an instance of cls represented by the dds message
        :param message: dict containing all fields of the class
        :return: object of type cls, constructed with the arguments from message
        """
        deser_dict = {}
        for name, tpe in cls.__init__.__annotations__.items():
            if inspect.isclass(tpe):
                if issubclass(tpe, np.ndarray):
                    deser_dict[name] = np.array(message[name]['array']).reshape(tuple(message[name]['shape']))
                elif issubclass(tpe, DDSTypedMsg):
                    deser_dict[name] = tpe.deserialize(message[name])
                elif issubclass(tpe, List):
                    deser_dict[name] = list(map(lambda d: tpe.__args__[0].deserialize(d), message[name]))
                else:
                    deser_dict[name] = message[name]
        return cls(**deser_dict)
