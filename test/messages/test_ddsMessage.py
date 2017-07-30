from src.messages.dds_message import DDSMessage
import numpy as np


class Foo(DDSMessage):
    def __init__(self, a: float, b: float):
        self._a = a
        self._b = b


class Voo(DDSMessage):
    def __init__(self, x: Foo, y: np.ndarray):
        self._x = x
        self._y = y


def test_serialize_dummyMsg_successful():
    f = Foo(2, 3)
    v = Voo(f, np.array([[.1, .2, 3], [11, 22, 33]]))
    v_ser = v.serialize()

    v_new = Voo.deserialize(v_ser)

    assert isinstance(v_new, DDSMessage)
