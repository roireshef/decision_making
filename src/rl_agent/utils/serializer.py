from enum import Enum

import numpy as np


class Serializer:
    @staticmethod
    def to_dict(elem, left_out_fields=None):
        """
        used to create the lcm message
        :param: left_out_fields: A list containing the fields we want to leave out while converting to dictionary
        :return: dict containing all the fields of the class
        """
        if left_out_fields is None:
            left_out_fields = []
        ser_dict = {}
        items = elem.items() if isinstance(elem, dict) else elem.__dict__.items()
        self_fields = {k: v for k, v in items if k not in left_out_fields}
        for key, val in self_fields.items():
            ser_dict[key] = Serializer._serialize_element(val)
        return {elem.__class__.__name__: ser_dict}

    @staticmethod
    def _serialize_element(elem):
        if issubclass(type(elem), np.ndarray):
            ser_elem = {'array': Serializer._serialize_element(elem.flat.__array__().tolist()),
                        'shape': list(elem.shape)}
        elif issubclass(type(elem), list):
            ser_elem = list(map(lambda x: Serializer._serialize_element(x), elem))
        elif issubclass(type(elem), Enum):
            ser_elem = str(elem)
        elif issubclass(type(elem), dict):
            ser_elem = {Serializer._serialize_element(k): Serializer._serialize_element(v) for k, v in elem.items()}
        elif getattr(elem, "__dict__", None):
            ser_elem = Serializer.to_dict(elem)
        else:
            ser_elem = elem
        return ser_elem

