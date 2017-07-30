from abc import ABCMeta, abstractmethod


class DDSMessage(metaclass=ABCMeta):
    def serialize(self)->dict:
        """
        used to create the dds message
        :return: dict containing all the fields of the class
        """
        return self.__dict__

    @classmethod
    def deserialize(cls, message: dict):
        """
        used to create an instance of cls represented by the dds message
        :param message: dict containing all fields of the class
        :return: object of type cls, constructed with the arguments from message
        """
        return cls(**message)
