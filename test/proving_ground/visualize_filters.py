import ast
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from decision_making.paths import Paths
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpaceContainer
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.action_space.road_sign_action_space import RoadSignActionSpace
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.data_objects import ActionType, RelativeLane
from decision_making.src.planning.behavioral.default_config import DEFAULT_ACTION_SPEC_FILTERING, \
    DEFAULT_STATIC_RECIPE_FILTERING, DEFAULT_DYNAMIC_RECIPE_FILTERING, DEFAULT_ROAD_SIGN_RECIPE_FILTERING
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from rte.python.logger.AV_logger import AV_Logger


def plot_filters_map(log_file_path: str):
    """
    Plot various graphs concerning localization, BP and TP outputs
    :param log_file_path: path to AV_Log_dm_main.log file
    :return: a showable matplotlib figure
    """
    file = open(log_file_path, 'r')
    f = plt.figure(1)
    gray_color = np.array([0.75, 0.75, 0.75])
    color_names = np.array(["gray", "r", "g", "b", "y", "k", "c", "m", "yellow", "lightblue", "peachpuff", "fuchsia", "papayawhip", "lightsalmon"])  # See https://matplotlib.org/examples/color/named_colors.html

    patches = []
    for idx, filter in enumerate(DEFAULT_ACTION_SPEC_FILTERING._filters + ['Passed']):
        patches.append(mpatches.Patch(color=color_names[idx], label=filter.__str__()))
    plt.legend(handles=patches)

    logger = AV_Logger.get_logger("Filters_visualizer")
    predictor = RoadFollowingPredictor(logger)
    action_space = ActionSpaceContainer(logger, [StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING),
                                                 DynamicActionSpace(logger, predictor, DEFAULT_DYNAMIC_RECIPE_FILTERING),
                                                 RoadSignActionSpace(logger, predictor, DEFAULT_ROAD_SIGN_RECIPE_FILTERING)])
    # TODO in the future remove this limitation of SAME_LANE
    limit_relative_lane = [RelativeLane.SAME_LANE]  # currently limit to SAME_LANE to make visualization easier.
    action_recipes = [recipe for recipe in action_space.recipes if recipe.relative_lane in limit_relative_lane]
    valid_idxs = [idx for idx, recipe in enumerate(action_space.recipes) if recipe.relative_lane in limit_relative_lane]
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
            filters_result = filters_result[valid_idxs]
            # filtering_map.append((timestamp, filters_result))
            plt.scatter(np.full(len(filters_result), timestamp), np.array(range(len(filters_result))),
                        c=color_names[filters_result], linestyle='None')
            # draw None actions with gray color which is not part of the color palette
            none_actions = np.where(filters_result == 0)[0]
            plt.scatter(np.full(len(none_actions), timestamp), none_actions,
                        c=np.tile(gray_color, len(none_actions)).reshape(-1, 3), linestyle='None')

        if 'Chosen behavioral action recipe' in text:
            recipe_str = text.split('Chosen behavioral action recipe')[1].split('Recipe: ')[1].replace("<", "'<").replace(">", ">'")
            recipe_dict = ast.literal_eval(recipe_str.split(' (ego_timestamp: ')[0])
            time = float(recipe_str.split(' (ego_timestamp: ')[1][:-2])

            # find chosen recipe
            recipe_type = int(recipe_dict['action_type'].split(':')[1].replace('>',''))
            aggressiveness = int(recipe_dict['aggressiveness'].split(':')[1].replace('>',''))
            relative_lane = int(recipe_dict['relative_lane'].split(':')[1].replace('>',''))
            if recipe_type == ActionType.FOLLOW_VEHICLE.value:
                relative_lon = int(recipe_dict['relative_lon'].split(':')[1].replace('>',''))

                # map it
                chosen_recipe_idx = [idx for idx, recipe in enumerate(action_recipes) if
                                     recipe.action_type == ActionType.FOLLOW_VEHICLE and
                                     recipe.aggressiveness.value == aggressiveness and
                                     recipe.relative_lane.value == relative_lane and
                                     recipe.relative_lon.value == relative_lon]
            elif recipe_type == ActionType.FOLLOW_LANE.value:
                velocity = recipe_dict['velocity']
                # map it
                chosen_recipe_idx = [idx for idx, recipe in enumerate(action_recipes) if
                                     recipe.action_type == ActionType.FOLLOW_LANE and
                                     recipe.aggressiveness.value == aggressiveness and
                                     recipe.relative_lane.value == relative_lane and
                                     np.isclose(recipe.velocity, velocity, atol=0.01)]
            elif recipe_type == ActionType.FOLLOW_ROAD_SIGN.value:
                relative_lon = int(recipe_dict['relative_lon'].split(':')[1].replace('>',''))

                # map it
                chosen_recipe_idx = [idx for idx, recipe in enumerate(action_recipes) if
                                     recipe.action_type == ActionType.FOLLOW_ROAD_SIGN and
                                     recipe.aggressiveness.value == aggressiveness and
                                     recipe.relative_lane.value == relative_lane and
                                     recipe.relative_lon.value == relative_lon]
            else:
                err_msg = "Unknown action %s" % recipe_dict
                raise AssertionError(err_msg)

            # plot with black x
            plt.scatter(np.array([time]), np.array([chosen_recipe_idx]), c='k', linestyle='None', marker='x')

    plt.xlabel('time [s]')
    plt.ylabel('action')
    return f


if __name__ == "__main__":
    # Enter path of log file to analyze here:
    file_path = '%s/../logs/AV_Log_dm_main.log' % Paths.get_repo_path()
    f = plot_filters_map(file_path)
    plt.show(f)
