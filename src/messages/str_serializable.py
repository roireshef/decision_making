from enum import Enum
import numpy as np


class StrSerializable:
    def to_dict(self):
        """
        used to create the dds message
        :return: dict containing all the fields of the class
        """
        ser_dict = {}
        self_fields = {k: v for k, v in self.__dict__.items() if k[0] != '_'}
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
        used to create the dds message
        :return: dict containing all the fields of the class
        """
        return str(self.to_dict())