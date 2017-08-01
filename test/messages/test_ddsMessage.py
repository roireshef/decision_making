import sys
import numpy as np

if sys.version_info > (3, 0):
    from test.messages.typed_messages_fixture import *
else:
    from test.messages.nontyped_messages_fixture import *


def test_serialize_dummyMsg_successful():
    f = Foo(2, 3)
    v = Voo(f, np.array([[.1, .2, 3], [11, 22, 33]]))
    v_ser = v.serialize()

    v_new = Voo.deserialize(v_ser)

    assert isinstance(v_new, Voo)
    assert isinstance(v_new.x, Foo)
    assert isinstance(v_new.y, np.ndarray)