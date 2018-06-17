from enum import Enum
import numpy as np


class StrSerializable:
    def to_dict(self, left_out_fields=None):
        """
        used to create the lcm message
        :param: left_out_fields: A list containing the fields we want to leave out while converting to dictionary
        :return: dict containing all the fields of the class
        """
        if left_out_fields is None:
            left_out_fields = []
        ser_dict = {}
        self_fields = {k: v for k, v in self.__dict__.items() if (k[0] != '_' and k not in left_out_fields)}
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
