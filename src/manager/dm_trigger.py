from abc import ABC, abstractmethod
from enum import Enum
from rte.python.periodic_timer.periodic_timer import PeriodicTimer


class DmTriggerType(Enum):
    DM_TRIGGER_PERIODIC = 0


class DmTrigger(ABC):

    def __init__(self, callback):
        self.callback = callback

    @abstractmethod
    def is_active(self):
        pass

    @abstractmethod
    def activate(self):
        pass

    @abstractmethod
    def deactivate(self):
        pass


class DmPeriodicTimerTrigger(DmTrigger):

    def __init__(self, callback, period):
        super().__init__(callback)
        self.is_active = False
        self.period = period
        if self.period > 0:
            self.timer = PeriodicTimer(self.period, self.callback)

    def is_active(self):
        return self.is_active

    def activate(self):
        if self.period > 0:
            self.is_active = True
            self.timer.start(run_in_thread=False)

    def deactivate(self):
        if self.is_active:
            self.timer.stop()
            self.is_active = False


