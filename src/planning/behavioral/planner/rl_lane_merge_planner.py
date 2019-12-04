from logging import Logger
import numpy as np
from decision_making.paths import Paths
from decision_making.src.exceptions import NoActionsLeftForBPError
from decision_making.src.global_constants import BP_JERK_S_JERK_D_TIME_WEIGHTS, EPS, \
    LANE_MERGE_ACTION_SPACE_MAX_VELOCITY, LANE_MERGE_ACTION_SPACE_VELOCITY_RESOLUTION, \
    LANE_MERGE_ACTION_SPACE_AGGRESSIVENESS_LEVEL
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.data_objects import StaticActionRecipe, ActionSpec, RelativeLane, \
    AggressivenessLevel
from decision_making.src.planning.behavioral.default_config import DEFAULT_STATIC_RECIPE_FILTERING
from decision_making.src.planning.behavioral.rl_models.dual_input_conv_model import DualInputConvModel
from decision_making.src.planning.behavioral.planner.base_planner import BasePlanner
from decision_making.src.planning.types import ActionSpecArray, FS_1D_LEN, FS_SX, FS_SV, FS_SA
from decision_making.src.planning.utils.kinematics_utils import KinematicUtils
from decision_making.src.planning.utils.numpy_utils import NumpyUtils
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.planning.behavioral.state.lane_merge_state import LaneMergeState
from decision_making.src.state.state import State
from ray.rllib.evaluation import SampleBatch
import torch
from gym.spaces.tuple import Tuple as GymTuple
from gym.spaces.box import Box
from ray.rllib.models.pytorch.model import TorchModel


class RL_LaneMergePlanner(BasePlanner):

    def __init__(self, logger: Logger):
        super().__init__(logger)
        self.predictor = RoadFollowingPredictor(logger)
        self.action_space = StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING)
        self.model = RL_LaneMergePlanner.load_model(Paths.get_policy_model())

    @staticmethod
    def load_model(model_path: str) -> TorchModel:
        """
        Load RL policy model from file
        :param model_path: model path
        :return: the loaded RL model
        """
        model_state_dict = torch.load(model_path)

        # TODO: create global constants for observation space initialization
        ego_box = Box(low=-np.inf, high=np.inf, shape=(1, FS_1D_LEN), dtype=np.float32)
        actors_box = Box(low=-np.inf, high=np.inf, shape=(2, 68), dtype=np.float32)
        obs_space = GymTuple((ego_box, actors_box))
        options = {"custom_options": {"hidden_size": 64}}
        model = DualInputConvModel(obs_space=obs_space, num_outputs=6, options=options)
        model.load_state_dict(model_state_dict)
        return model

    def _create_behavioral_state(self, state: State, route_plan: RoutePlan) -> LaneMergeState:
        return LaneMergeState.create_from_state(state, route_plan, self.logger)

    def _create_action_specs(self, lane_merge_state: LaneMergeState) -> ActionSpecArray:
        """
        Create action space compatible with the current RL checkpoint (for example 6 static actions
        with target velocities 0, 5, 10, 15, 20, 25 and STANDARD aggressiveness level).
        :param lane_merge_state: LaneMergeState that inherits from BehavioralGridState
        :return: array of action specs
        """
        s_0, v_0, a_0 = lane_merge_state.ego_fstate_1d
        v_T = np.arange(0, LANE_MERGE_ACTION_SPACE_MAX_VELOCITY + EPS, LANE_MERGE_ACTION_SPACE_VELOCITY_RESOLUTION)

        # time-jerk weights for a given aggressiveness level
        w_J, _, w_T = BP_JERK_S_JERK_D_TIME_WEIGHTS[LANE_MERGE_ACTION_SPACE_AGGRESSIVENESS_LEVEL]

        # specify static actions for all target velocities (without time limit and without acceleration limits)
        ds, T = KinematicUtils.specify_quartic_actions(w_T, w_J, v_0, v_T, a_0, action_horizon_limit=np.inf, acc_limits=None)

        # create action specs based on the above output
        aggr_level = AggressivenessLevel(LANE_MERGE_ACTION_SPACE_AGGRESSIVENESS_LEVEL)
        return np.array([ActionSpec(t=t, v=vt, s=s_0 + s, d=0,
                                    recipe=StaticActionRecipe(RelativeLane.SAME_LANE, vt, aggr_level))
                         for vt, t, s in zip(v_T, T, ds)])

    def _filter_actions(self, lane_merge_state: LaneMergeState, action_specs: ActionSpecArray) -> ActionSpecArray:
        """
        filter out actions that either are filtered by a regular action_spec filter (of the single_lane_planner)
        or don't enable to brake before the red line
        :param action_specs: array of ActionSpec (part of actions may be None)
        :return: array of ActionSpec of the original size, with None for filtered actions
        """
        return action_specs  # DON'T filter the actions

    def _evaluate_actions(self, lane_merge_state: LaneMergeState, route_plan: RoutePlan,
                          action_specs: ActionSpecArray) -> np.ndarray:
        """
        Given RL policy, current state and actions mask, return the actions cost by operating policy on the state.
        Filtered actions get the maximal cost 1.
        :param lane_merge_state: lane merge state
        :param action_specs: np.array of action specs
        :return: array of actions costs: the lower the better
        """
        action_mask = action_specs.astype(bool)
        if not action_mask.any():
            raise NoActionsLeftForBPError("All actions were filtered in BP. timestamp_in_sec: %f" %
                                          lane_merge_state.ego_state.timestamp_in_sec)

        # encode the state
        host_state, actors_state = LaneMergeState.encode_state(
            lane_merge_state.ego_fstate_1d, lane_merge_state.red_line_s_on_ego_gff, lane_merge_state.actors_states)
        encoded_state: GymTuple = (torch.from_numpy(host_state[np.newaxis, :]).float(),
                                   torch.from_numpy(actors_state[np.newaxis, :]).float())

        # call RL inference
        logits, _, values, _ = self.model._forward({SampleBatch.CUR_OBS: encoded_state})

        # convert logits to probabilities
        torch_probabilities = torch.nn.functional.softmax(logits, dim=1)
        probabilities = torch_probabilities.detach().numpy()[0]

        # mask filtered actions and return costs (1 - probability)
        costs = 1 - probabilities * action_mask.astype(int)
        return costs

    def _choose_action(self, lane_merge_state: LaneMergeState, action_specs: ActionSpecArray, costs: np.array) -> \
            ActionSpec:
        return action_specs[np.argmin(costs)]
