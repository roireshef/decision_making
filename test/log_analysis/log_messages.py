import ast
import inspect
import traceback
from builtins import Exception
from enum import Enum
from typing import List, Type

import numpy as np

from decision_making.src.exceptions import MsgDeserializationError, MsgSerializationError
from decision_making.src.global_constants import PUBSUB_MSG_IMPL


class LogMsg(object):
    @staticmethod
    def convert_message_to_dict(message: str) -> dict:
        """
        Convert message string from log to dictionary that can be deserialized to the message object
        :param message:
        :return:
        """
        return ast.literal_eval(message.replace('<', '\'<').replace('>', '>\''))

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
            if len(annotations) == 0:
                # Non typed message (Python 2)
                annotations = inspect.getmembers(class_type,
                                                 lambda a: not (isinstance(a, property)) and not (inspect.isroutine(a)))
                annotations = [a for a in annotations if not a[0].startswith('_')]
            for name, tpe in annotations:
                if issubclass(tpe, np.ndarray):
                    deser_dict[name] = np.array(message[name]['array']).reshape(tuple(message[name]['shape']))
                elif issubclass(tpe, Enum):
                    deser_dict[name] = tpe[message[name]['name']]
                elif issubclass(tpe, List):
                    deser_dict[name] = list(map(lambda d: LogMsg.deserialize(class_type=tpe.__args__[0], message=d),
                                                message[name]['iterable']))
                elif issubclass(tpe, PUBSUB_MSG_IMPL):
                    deser_dict[name] = LogMsg.deserialize(class_type=tpe, message=message[name])
                else:
                    deser_dict[name] = message[name]
            return class_type(**deser_dict)
        except Exception as e:
            raise MsgDeserializationError("MsgDeserializationError error: could not deserialize into " +
                                          class_type.__name__ + " from " + str(message) + ":\n" + str(
                traceback.print_exc()))
