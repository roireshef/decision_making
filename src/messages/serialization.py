import inspect
from enum import Enum
import numpy as np


# The key of static member that determines which fields will be left out when serializing the class
SERIALIZABLE_LEFT_OUT_FIELDS_KEY = 'left_out_fields'


class StrSerializable:
    def get_internal_left_out_fields(self):
        """
        Returns the left out fields that shouldn't be serialized
         (determined by the proper static member SERIALIZABLE_LEFT_OUT_FIELDS_KEY)
        :return: list of left out fields
        """
        serialization_annotation = inspect.getmembers(self.__class__,
                                                      lambda a: not (isinstance(a, property)) and not (
                                                          inspect.isroutine(a)))
        internal_left_out_fields = [a[1] for a in serialization_annotation if a[0] == SERIALIZABLE_LEFT_OUT_FIELDS_KEY]
        return internal_left_out_fields

    def to_dict(self, left_out_fields=None):
        """
        used to create the lcm message
        :param: left_out_fields: A list containing the fields we want to leave out while converting to dictionary
        :return: dict containing all the fields of the class
        """
        internal_left_out_fields = self.get_internal_left_out_fields()
        if left_out_fields is None:
            left_out_fields = []
        ser_dict = {}
        self_fields = {k: v for k, v in self.__dict__.items() if (
                k not in left_out_fields and k not in internal_left_out_fields)}
        for key, val in self_fields.items():
            ser_dict[key] = StrSerializable._serialize_element(val)
        return ser_dict

    @staticmethod
    def _serialize_element(elem):
        if issubclass(type(elem), np.ndarray):
            ser_elem = {'array': elem.flat.__array__().tolist(), 'shape': list(elem.shape)}
        elif issubclass(type(elem), list):
            ser_elem = {'iterable': list(map(lambda x: StrSerializable._serialize_element(x) if issubclass(type(x), list) else x.to_dict(), elem))}
        elif issubclass(type(elem), Enum):
            ser_elem = {'name': elem.name}
        elif issubclass(type(elem), StrSerializable):
            ser_elem = elem.to_dict()
        else:
            ser_elem = elem
        return ser_elem

    def __str__(self):
        """
        used to create the lcm message
        :return: dict containing all the fields of the class
        """
        return str(self.to_dict())


# PubSub message class implementation for all DM messages
PUBSUB_MSG_IMPL = StrSerializable