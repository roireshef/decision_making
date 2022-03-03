from decision_making.src.rl_agent.environments.action_space.common.data_objects import ConstantAccelerationRLActionRecipe
from decision_making.src.rl_agent.environments.action_space.common.lateral_mixins import CommitLaneChangeMixin
from decision_making.src.rl_agent.environments.action_space.common.longitudinal_mixins import AccelerationCommandsMixin
from decision_making.src.rl_agent.environments.action_space.trajectory_based_action_space_adapter import \
    TrajectoryBasedActionSpaceAdapter
from decision_making.src.rl_agent.global_types import LateralDirection


class ChangeLaneAtConstantSpeedActionSpaceAdapter(CommitLaneChangeMixin, AccelerationCommandsMixin,
                                                  TrajectoryBasedActionSpaceAdapter):
    """
    Implements action space of three actions:
    1. Initiate Right lane change (will be followed by (2))
    2. NOOP - either keep lane or keep changing lanes with no speed change
    3. Initiate Left lane change (will be followed by (2))
    """
    def __init__(self, action_space_params) -> None:
        super().__init__(action_space_params,
                         [ConstantAccelerationRLActionRecipe(lat_dir, 0.0) for lat_dir in LateralDirection.__iter__()])
