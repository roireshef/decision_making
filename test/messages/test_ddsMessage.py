import sys
import numpy as np

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