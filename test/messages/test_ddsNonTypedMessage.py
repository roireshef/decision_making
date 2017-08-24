import numpy as np

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


def test_deserialize_validEnum_successful():
    m = Moo(TrajectoryPlanningStrategy.PARKING)

    m_ser = m.serialize()

    assert m_ser.keys().__contains__('strategy') and m_ser['strategy']['name'] == 'PARKING'

    m_deser = Moo.deserialize(m_ser)

    assert TrajectoryPlanningStrategy[m_deser.strategy.name].value == m_deser.strategy.value