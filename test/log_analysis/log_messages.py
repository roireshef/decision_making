import ast
import inspect
import traceback
from builtins import Exception
from enum import Enum
from typing import List, Type

import numpy as np

from decision_making.src.exceptions import MsgDeserializationError, MsgSerializationError


class LogMsg(object):

    @staticmethod
    def convert_message_to_dict(message: str) -> dict:
        """
        Convert message string from log to dictionary that can be deserialized to the message object
        :param message:
        :return:
        """
        return ast.literal_eval(message)

    @staticmethod
    def pubsub_serialize(obj: object) -> dict:
        """
        Serializes the object to PubSub dictionary
        :return: dict containing all the fields of the class
        """
        self_dict = obj.__dict__
        ser_dict = {}

        # enumerate all fields (and their types) in the constructor
        for name, tpe in obj.__init__.__annotations__.items():
            try:
                if issubclass(tpe, np.ndarray):
                    ser_dict[name] = {'array': self_dict[name].flat.__array__().tolist(),
                                      'shape': list(self_dict[name].shape)}
                elif issubclass(tpe, list):
                    ser_dict[name] = list(map(lambda x: x.serialize(), self_dict[name]))
                elif issubclass(tpe, Enum):
                    ser_dict[name] = self_dict[name].name  # save the name of the Enum's value (string)
                elif inspect.isclass(tpe) and issubclass(tpe, LogMsg):
                    ser_dict[name] = self_dict[name].serialize()
                # if the member type in the constructor is a primitive - copy as is
                else:
                    ser_dict[name] = self_dict[name]
            except Exception as e:
                raise MsgSerializationError("MsgSerializationError error: could not serialize " +
                                            str(self_dict[name]) + " into " + str(tpe) + ":\n" + str(e.__traceback__))
        return ser_dict

    @staticmethod
    def deserialize(class_type: Type, message: dict):
        """
        Creates an instance of cls represented by the log message
        :param message: dict containing all fields of the class
        :return: object of type cls, constructed with the arguments from message
        """
        try:
            deser_dict = {}
            annotations = class_type.__init__.__annotations__.items()
            for name, tpe in annotations:
                if issubclass(tpe, np.ndarray):
                    deser_dict[name] = np.array(message[name]['array']).reshape(tuple(message[name]['shape']))
                elif issubclass(tpe, Enum):
                    deser_dict[name] = tpe[message[name]['name']]
                elif issubclass(tpe, List):
                    deser_dict[name] = list(map(lambda d: tpe.__args__[0].deserialize(d), message[name]['iterable']))
                elif issubclass(tpe, LogMsg):
                    deser_dict[name] = LogMsg.deserialize(tpe, message[name])
                else:
                    deser_dict[name] = message[name]
            return class_type(**deser_dict)
        except Exception as e:
            raise MsgDeserializationError("MsgDeserializationError error: could not deserialize into " +
                                          class_type.__name__ + " from " + str(message) + ":\n" + str(traceback.print_exc()))
