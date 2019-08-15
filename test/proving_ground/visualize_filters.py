import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from decision_making.paths import Paths
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpaceContainer
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.data_objects import ActionType
from decision_making.src.planning.behavioral.default_config import DEFAULT_ACTION_SPEC_FILTERING, \
    DEFAULT_STATIC_RECIPE_FILTERING, DEFAULT_DYNAMIC_RECIPE_FILTERING
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from rte.python.logger.AV_logger import AV_Logger


def plot_filters_map(log_file_path: str):
    """
    Plot which actions were filtered by which action_spec filters
    :param log_file_path: path to AV_Log_dm_main.log file
    :return: a showable matplotlib figure
    """
    file = open(log_file_path, 'r')
    f = plt.figure(1)
    colors_num = len(DEFAULT_ACTION_SPEC_FILTERING._filters) + 2
    map_idx_to_color = np.array([4, 6, 2, 0, 8, 7, 5, 1, 3]) / colors_num
    gray_color = np.array([0.75, 0.75, 0.75])

    patches = []
    for idx, filter in enumerate(DEFAULT_ACTION_SPEC_FILTERING._filters + ['Passed']):
        color = plt.cm.hsv(map_idx_to_color[idx]) if idx > 0 else gray_color
        patches.append(mpatches.Patch(color=color, label=filter.__str__()))
    plt.legend(handles=patches)

    logger = AV_Logger.get_logger("Filters_visualizer")
    predictor = RoadFollowingPredictor(logger)
    action_space = ActionSpaceContainer(logger, [StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING),
                                                 DynamicActionSpace(logger, predictor, DEFAULT_DYNAMIC_RECIPE_FILTERING)])
    action_recipes = action_space.recipes
    y_values = [(recipe.action_type.__str__().split('_')[1][:4], recipe.relative_lane.__str__().split('.')[1][:4],
                 recipe.aggressiveness.__str__().split('.')[1][:4],
                 '%.1f' % recipe.velocity if recipe.action_type == ActionType.FOLLOW_LANE else '')
                for recipe in action_recipes]
    y_axis = np.arange(len(action_recipes))
    plt.yticks(y_axis, y_values)

    while True:
        text = file.readline()
        if not text:
            break

        if 'Filtering_map' in text:
            colon_str = text.split('timestamp_in_sec ')[1].split(':')
            timestamp = float(colon_str[0])
            filters_result = np.array(list(map(int, colon_str[1].replace('array([', '').replace('])', '').split(', '))))
            # filtering_map.append((timestamp, filters_result))
            colors = plt.cm.hsv(map_idx_to_color[filters_result])
            plt.scatter(np.full(len(filters_result), timestamp), np.array(range(len(filters_result))),
                        c=colors, linestyle='None')
            # draw None actions with gray color which is not part of the color palette
            none_actions = np.where(filters_result == 0)[0]
            plt.scatter(np.full(len(none_actions), timestamp), none_actions,
                        c=np.tile(gray_color, len(none_actions)).reshape(-1, 3), linestyle='None')

    plt.xlabel('time [s]')
    plt.ylabel('action')
    return f


if __name__ == "__main__":
    # Enter path of log file to analyze here:
    file_path = '%s/../logs/AV_Log_dm_main.log' % Paths.get_repo_path()
    f = plot_filters_map(file_path)
    plt.show(f)
