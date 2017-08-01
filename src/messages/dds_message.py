from abc import ABCMeta, abstractmethod

import six


@six.add_metaclass(ABCMeta)
class DDSMsg:
    @abstractmethod
    def serialize(self):
        pass

    @classmethod
    @abstractmethod
    def deserialize(cls, message):
        pass




#
# class DDSMessage(metaclass=ABCMeta):
#     def serialize(self)->dict:
#         """
#         used to create the dds message
#         :return: dict containing all the fields of the class
#         """
#         complex_dict = self.__dict__.copy()
#         for key, val in complex_dict.items():
#             # handle complex types
#             if isinstance(val, np.ndarray):
#                 complex_dict[key] = {'array': val.flat.__array__().tolist(), 'shape': list(val.shape)}
#             elif isinstance(val, DDSMessage):
#                 complex_dict[key] = val.serialize()
#         return complex_dict
#
#     @classmethod
#     def deserialize(cls, message: dict):
#         """
#         used to create an instance of cls represented by the dds message
#         :param message: dict containing all fields of the class
#         :return: object of type cls, constructed with the arguments from message
#         """
#         message_copy = message.copy()
#         for name, type in cls.__init__.__annotations__.items():
#             if 'numpy.ndarray' in str(type):
#                 message_copy[name] = np.array(message_copy[name]['array']).reshape(tuple(message_copy[name]['shape']))
#             elif isinstance(type, ABCMeta):
#                 real_type = type(type.__name__, '')
#                 if isinstance(real_type, DDSMessage):
#                     message_copy[name] = real_type.deserialize(message_copy[name])
#         return cls(**message_copy)




