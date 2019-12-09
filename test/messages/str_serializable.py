from decision_making.src.messages.serialization import PUBSUB_MSG_IMPL
from typing import List


class SimpleStruct(PUBSUB_MSG_IMPL):
    x = int
    y = str

    def __init__(self, x: int, y: str):
        self.y = y
        self.x = x


class MessageWithNestedLists(PUBSUB_MSG_IMPL):
    ll = List[List[SimpleStruct]]

    def __init__(self, ll: List[List[SimpleStruct]]):
        self.ll = ll


def test_toDict_messageWithNestedLists_terminatesSuccessfully():
    ll = MessageWithNestedLists([[SimpleStruct(1, 'one'), SimpleStruct(2, 'two')],[SimpleStruct(3, 'three')]])
    print(ll.__str__())

    assert True
