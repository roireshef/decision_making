from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable

from rte.python.periodic_timer.periodic_timer import PeriodicTimer
from decision_making.src.manager.trigger_exceptions import DmTriggerActivationException


class DmTriggerType(Enum):
    DM_TRIGGER_NONE = 0  # for modules without a trigger
    DM_TRIGGER_PERIODIC = 1


class DmTrigger(ABC):
    """
    The abstract class for triggers
    """
    def __init__(self):
        pass

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
        super().__init__()

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

    def __init__(self, callback: Callable[[], None], process_name: str, period: float):
        super().__init__()
        self._callback = callback
        self._is_active = False
        self._period = period
        if self._period > 0:
            self._timer = PeriodicTimer(self._period, self._callback, name=process_name)
        else:
            raise ValueError('invalid period ({}) set for DmPeriodicTimerTrigger'.format(period))

    def is_active(self) -> bool:
        return self._is_active

    def activate(self):
        if not self._is_active:
            self._is_active = True
            self._timer.start()
        else:
            raise DmTriggerActivationException('trying to activate an already active DmPeriodicTimerTrigger')

    def deactivate(self):
        if self._is_active:
            self._timer.stop()
            self._is_active = False
        else:
            raise DmTriggerActivationException('trying to deactivate an already inactive DmPeriodicTimerTrigger')

    def period(self):
        return self._period