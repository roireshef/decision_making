import sys
import numpy as np
import py, pytest


if sys.version_info > (3, 0):
    from decision_making.test.messages.typed_messages_fixture import *
else:
    from decision_making.test.messages.nontyped_messages_fixture import *


def test_serialize_dummyMsg_successful():
    f = Foo(2, 3)
    v = Voo(f, np.array([[.1, .2, 3], [11, 22, 33]]))
    w = Woo(list((v, v)))
    w_ser = w.serialize()

    w_new = Woo.deserialize(w_ser)

    assert isinstance(w_new, Woo)
    assert isinstance(w_new.l, list)
    assert isinstance(w_new.l[0].y, np.ndarray)

def test_serialize_dummyWrongFieldsMsg_throwsError():
    f = Foo(2, 3)
    v = Voo(f, np.array([[.1, .2, 3], [11, 22, 33]]))
    w = Woo(list((f, v)))

    try:
        w_ser = w.serialize()
        w_new = Woo.deserialize(w_ser)
        pytest.fail("An exception was meant to be thrown but nothing was actually thrown")
    except Exception:
        assert True
