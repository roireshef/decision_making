import pytest

from decision_making.src.exceptions import MsgDeserializationError, MsgSerializationError
from decision_making.test.messages.typed_messages_fixture import *


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
    f = Foo(2.0, 3.0)
    v_invalid = Voo(2, np.array([[.1, .2, 3], [11, 22, 33]]))
    with pytest.raises(MsgSerializationError,
                       message="Trying to serialize wrong class-types passed without an exception"):
        v_ser = v_invalid.serialize()

    v_valid = Voo(f, np.array([[.1, .2, 3], [11, 22, 33]]))
    v_ser_invalid = v_valid.serialize()
    v_ser_invalid['x'] = 2.0
    with pytest.raises(MsgDeserializationError,
                       message="Trying to deserialize wrong class-types passed without an exception"):
        v_new = Voo.deserialize(v_ser_invalid)


def test_deserialize_validEnum_successful():
    m = Moo(TrajectoryPlanningStrategy.PARKING)

    m_ser = m.serialize()

    assert m_ser.keys().__contains__('strategy') and m_ser['strategy'] == 'PARKING'

    m_deser = Moo.deserialize(m_ser)

    assert TrajectoryPlanningStrategy[m_deser.strategy.name].value == m_deser.strategy.value


def test_deserialize_invalidEnum_throwsError():
    m_ser = {'strategy': 'LANDING_ON_THE_MOON'}
    with pytest.raises(MsgDeserializationError, message="Trying to deserialize wrong enum value w/o exception"):
        Moo.deserialize(m_ser)
