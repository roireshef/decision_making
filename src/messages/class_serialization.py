import ast
import inspect
import traceback
from builtins import Exception
from enum import Enum
from typing import List, Type, Tuple

import numpy as np

from decision_making.src.exceptions import MsgDeserializationError
from decision_making.src.messages.serialization import PUBSUB_MSG_IMPL
from decision_making.src.messages.serialization import SERIALIZABLE_LEFT_OUT_FIELDS_KEY

MEMBERS_REMAPPING_KEY = 'members_remapping'


class ClassSerializer(object):
    @staticmethod
    def get_annotations(class_type: Type) -> List[Tuple[str, Type]]:
        """
        Get annotations of certain class. Both typed and non-typed (Python 2,3 compatible)
        :param class_type:
        :return:
        """
        annotations = class_type.__init__.__annotations__.items()
        if len(annotations) == 0:
            # Non typed message (Python 2)
            annotations = inspect.getmembers(class_type,
                                             lambda a: not (isinstance(a, property)) and not (inspect.isroutine(a)))
            annotations = [a for a in annotations if
                           not a[0].startswith('__') and not a[0] == MEMBERS_REMAPPING_KEY and not a[
                                                                                                       0] == SERIALIZABLE_LEFT_OUT_FIELDS_KEY]

        return annotations

    @staticmethod
    def get_members_remapping(class_type: Type) -> (List[Tuple[str, str]], List[str]):
        """
        Gets a dictionary of members to be remapped to a new name.
         (For cases when the name in the constructor is mapped to a different member name)
        :param class_type: class type
        :return: dictionary that maps old member to new member name, dictionary of old member keys
        """
        serialization_annotation = inspect.getmembers(class_type,
                                                      lambda a: not (isinstance(a, property)) and not (
                                                          inspect.isroutine(a)))

        members_remapping = [a[1] for a in serialization_annotation if a[0] == MEMBERS_REMAPPING_KEY]
        if len(members_remapping) > 0:
            members_remapping = members_remapping[0]
            members_remapping_keys = members_remapping.keys()
        else:
            members_remapping_keys = []

        return members_remapping, members_remapping_keys

    @staticmethod
    def convert_message_to_dict(message: str) -> dict:
        """
        Convert message string from log to dictionary that can be deserialized to the message object
        :param message:
        :return:
        """
        return ast.literal_eval(message)

    @staticmethod
    def deserialize(class_type: Type, message: dict, allow_none_objects: bool = False):
        """
        Creates an instance of cls represented by the log message
        :param allow_none_objects: allow some objects to be none instead of their regular type
        :param message: dict containing all fields of the class
        :return: object of type cls, constructed with the arguments from message
        """
        try:
            deser_dict = {}

            annotations = ClassSerializer.get_annotations(class_type=class_type)
            members_remapping, members_remapping_keys = ClassSerializer.get_members_remapping(class_type=class_type)

            for name, tpe in annotations:

                # Remap to the new name
                if name in members_remapping_keys:
                    target_name = members_remapping[name]
                else:
                    target_name = name

                if allow_none_objects:
                    if message[name] is None:
                        deser_dict[target_name] = None
                        continue

                if issubclass(tpe, np.ndarray):
                    deser_dict[target_name] = np.array(message[name]['array']).reshape(tuple(message[name]['shape']))
                elif issubclass(tpe, Enum):
                    deser_dict[target_name] = tpe[message[name]['name']]
                elif issubclass(tpe, List):
                    deser_dict[target_name] = list(
                        map(lambda d: ClassSerializer.deserialize(class_type=tpe.__args__[0], message=d,
                                                                  allow_none_objects=allow_none_objects),
                            message[name]['iterable']))
                elif issubclass(tpe, PUBSUB_MSG_IMPL):
                    deser_dict[target_name] = ClassSerializer.deserialize(class_type=tpe, message=message[name],
                                                                          allow_none_objects=allow_none_objects)
                else:
                    deser_dict[target_name] = message[name]
            return class_type(**deser_dict)
        except Exception as e:
            raise MsgDeserializationError("MsgDeserializationError error: could not deserialize into " +
                                          class_type.__name__ + " from " + str(message) + ":\n" + str(
                traceback.print_exc()))
