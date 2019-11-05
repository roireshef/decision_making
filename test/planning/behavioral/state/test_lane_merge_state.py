import numpy as np
from decision_making.src.planning.behavioral.planner.RL_lane_merge_planner import RL_LaneMergePlanner, CHECKPOINT_PATH
from gym.spaces.tuple_space import Tuple as GymTuple
from ray.rllib.evaluation import SampleBatch
from decision_making.src.planning.behavioral.state.lane_merge_state import LaneMergeState
from decision_making.test.planning.behavioral.behavioral_state_fixtures import state_with_objects_before_merge
from decision_making.test.planning.custom_fixtures import route_plan_1_2_3
from logging import Logger



def test_laneMergeState_loadModelAndForward_success(state_with_objects_before_merge, route_plan_1_2_3):

    model = RL_LaneMergePlanner.load_model(CHECKPOINT_PATH)

    logger = Logger("")
    lane_merge_state = LaneMergeState.create_from_state(state_with_objects_before_merge, route_plan_1_2_3, logger)
    encoded_state: GymTuple = lane_merge_state.encode_state_for_RL()
    logits, _, values, _ = model._forward({SampleBatch.CUR_OBS: encoded_state}, [])
    chosen_action_idx = np.argmax(logits.detach().numpy())
    assert chosen_action_idx == 0
