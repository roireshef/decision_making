from typing import NamedTuple, Optional


class SimCommand(NamedTuple):
    target_speed: Optional[float]     # [m/s] SUMO Controller acceleration command
    lateral_command: float  # [m] SUMO Controller lateral difference

    @classmethod
    def do_nothing(cls):
        return cls(None, .0)

