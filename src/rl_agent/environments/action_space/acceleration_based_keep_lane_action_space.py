from decision_making.src.rl_agent.environments.action_space.common.data_objects import ConstantAccelerationRLActionRecipe
from decision_making.src.rl_agent.environments.action_space.common.lateral_mixins import OnlyKeepLaneMixin
from decision_making.src.rl_agent.environments.action_space.common.longitudinal_mixins import AccelerationCommandsMixin
from decision_making.src.rl_agent.environments.action_space.trajectory_based_action_space_adapter import \
    TrajectoryBasedActionSpaceAdapter
from decision_making.src.rl_agent.global_types import LateralDirection


class AccelerationBasedKeepLaneActionSpaceAdapter(OnlyKeepLaneMixin, AccelerationCommandsMixin,
                                                  TrajectoryBasedActionSpaceAdapter):
    """
    Implements action space for actions that keep lane and apply acceleration/deceleration for some horizon. By
    including the value 0.0 in the list of accelerations you introduce a NOOP action. Note: when combined with
    ChangeLaneAtConstantSpeedActionSpaceAdapter, there is no need to have a NOOP action here!
    """
    def __init__(self, action_space_params) -> None:
        action_recipes = [ConstantAccelerationRLActionRecipe(lateral_dir=LateralDirection.SAME, acceleration=acc)
                          for acc in action_space_params['ACCELERATIONS']]
        super().__init__(action_space_params, action_recipes)
