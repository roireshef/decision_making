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





