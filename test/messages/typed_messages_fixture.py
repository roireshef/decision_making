from decision_making.src.messages.dds_typed_message import DDSTypedMsg
import numpy as np
from typing import List


class Foo(DDSTypedMsg):
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b


class Voo(DDSTypedMsg):
    def __init__(self, x: Foo, y: np.ndarray):
        self.x = x
        self.y = y


class Woo(DDSTypedMsg):
    def __init__(self, l: List[Voo]):
        self.l = l