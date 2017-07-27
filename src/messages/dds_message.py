from abc import ABCMeta
import numpy as np


class DDSMessage(metaclass=ABCMeta):
    def serialize(self)->dict:
        """
        used to create the dds message
        :return: dict containing all the fields of the class
        """
        complex_dict = {k[1:]: v for k, v in self.__dict__.items()}
        for key, val in complex_dict.items():
            if isinstance(val, np.ndarray):
                complex_dict[key] = {'array': val.__array__(), 'shape': val.shape}
            if isinstance(val, DDSMessage):
                complex_dict[key] = val.serialize()
        return complex_dict

    @classmethod
    def deserialize(cls, message: dict):
        """
        used to create an instance of cls represented by the dds message
        :param message: dict containing all fields of the class
        :return: object of type cls, constructed with the arguments from message
        """
        message_copy = message.copy()
        for name, type in cls.__init__.__annotations__.items():
            if 'numpy.ndarray' in str(type):
                message_copy[name] = np.array(message_copy[name]['array']).reshape(message_copy[name]['shape'])
            elif isinstance(type, ABCMeta):
                real_type = type(type.__name__, '')
                if isinstance(real_type, DDSMessage):
                    message_copy[name] = real_type.deserialize(message_copy[name])
        return cls(**message_copy)