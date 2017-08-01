from src.messages.dds_nontyped_message import DDSNonTypedMsg


class Foo(DDSNonTypedMsg):
    def __init__(self, a, b):
        self.a = a
        self.b = b


class Voo(DDSNonTypedMsg):
    def __init__(self, x, y):
        self.x = x
        self.y = y