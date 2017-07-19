from abc import ABCMeta, abstractmethod


class DdsMessage(metaclass=ABCMeta):
    @abstractmethod
    def serialize(self)->dict:
        """
        used to create the dds message
        :return: dict containing all the fields of the class
        """
        pass

    @abstractmethod
    def deserialize(self, message: dict)->None:
        """
        used to populate the fields of the class given a dds message dict
        :param message: dict containing all fields of the class
        :return: void
        """
        pass
