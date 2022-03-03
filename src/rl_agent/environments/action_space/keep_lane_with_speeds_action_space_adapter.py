from itertools import product

import numpy as np
from decision_making.src.planning.utils.numpy_utils import UniformGrid
from decision_making.src.rl_agent.environments.action_space.common.data_objects import \
    LateralOffsetTerminalVelocityActionRecipe
from decision_making.src.rl_agent.environments.action_space.common.lateral_mixins import OnlyKeepLaneMixin
from decision_making.src.rl_agent.environments.action_space.common.longitudinal_mixins import TerminalVelocityMixin
from decision_making.src.rl_agent.environments.action_space.trajectory_based_action_space_adapter import \
    TrajectoryBasedActionSpaceAdapter
from decision_making.src.rl_agent.global_types import LateralDirection


class KeepLaneWithSpeedActionSpaceAdapter(OnlyKeepLaneMixin, TerminalVelocityMixin,
                                          TrajectoryBasedActionSpaceAdapter):
    RECIPE_CLS = LateralOffsetTerminalVelocityActionRecipe

    def __init__(self, action_space_params) -> None:
        """ Implements Action Space of target velocities with lane centering """

        velocity_grid = UniformGrid(limits=np.array([action_space_params['MIN_VELOCITY'],
                                                     action_space_params['MAX_VELOCITY']]),
                                    resolution=action_space_params['VELOCITY_RESOLUTION'])
        super().__init__(action_space_params,
                         [LateralOffsetTerminalVelocityActionRecipe(LateralDirection.SAME, v, agg)
                          for v, agg in product(velocity_grid.array,
                                                action_space_params['AGGRESSIVENESS_LEVELS'])])

