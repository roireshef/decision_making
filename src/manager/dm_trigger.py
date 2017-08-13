from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable
from rte.python.periodic_timer.periodic_timer import PeriodicTimer


class DmTriggerType(Enum):
    DM_TRIGGER_NONE = 0 # for modules without a trigger
    DM_TRIGGER_PERIODIC = 1


class DmTrigger(ABC):

    def __init__(self, callback):
        self.callback = callback

    @abstractmethod
    def is_active(self) -> bool:
        pass

    @abstractmethod
    def activate(self):
        pass

    @abstractmethod
    def deactivate(self):
        pass


class DmNullTrigger(DmTrigger):
    """
    This trigger does nothing, it is for modules that don't need a trigger
    """

    def __init__(self):
        pass

    def is_active(self) -> bool:
        return False

    def activate(self):
        pass

    def deactivate(self):
        pass


class DmPeriodicTimerTrigger(DmTrigger):
    """
    This trigger will call the given callback according to the given period
    """

    def __init__(self, callback: Callable[[None], None], period: float):
        super().__init__(callback)
        self.is_active = False
        self.period = period
        if self.period > 0:
            self.timer = PeriodicTimer(self.period, self.callback)
        else:
            raise ValueError('invalid period ({}) set for DmPeriodicTimerTrigger'.format(period))

    def is_active(self) -> bool:
        return self.is_active

    def activate(self):
        if self.period > 0:
            self.is_active = True
            self.timer.start(run_in_thread=False)

    def deactivate(self):
        if self.is_active:
            self.timer.stop()
            self.is_active = False


