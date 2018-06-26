import inspect
from enum import Enum
import numpy as np

from decision_making.src.global_constants import SERIALIZABLE_LEFT_OUT_FIELDS_KEY


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
            if issubclass(type(val), np.ndarray):
                ser_dict[key] = {'array': val.flat.__array__().tolist(), 'shape': list(val.shape)}
            elif issubclass(type(val), list):
                ser_dict[key] = {'iterable': list(map(lambda x: x.to_dict(), val))}
            elif issubclass(type(val), Enum):
                ser_dict[key] = {'name': val.name}
            elif issubclass(type(val), StrSerializable):
                ser_dict[key] = val.to_dict()
            else:
                ser_dict[key] = val
        return ser_dict

    def __str__(self):
        """
        used to create the lcm message
        :return: dict containing all the fields of the class
        """
        return str(self.to_dict())
