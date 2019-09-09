import json

import matplotlib.pyplot as plt
from decision_making.paths import Paths
from decision_making.test.proving_ground import visualize_logs
from matplotlib.backends.backend_pdf import PdfPages


if __name__ == "__main__":
    path = Paths.get_repo_path()
    # Enter path of json file to analyze here:
    json_filename = path + '/../simulation/ground_truth_data/sim_metadata/run_info_file_2019_09_08_15_56_09_054772.json'
    with open(json_filename, 'r') as f:
        batch_data = f.read()
        batch_data = "{" + batch_data.replace("\n", "").replace("\'", "")[0:-1] + "}"
        dump_file = path + "/../logs/batch_" + json_filename.split("run_info_file_")[1].replace("json", "pdf")
        json_data = json.loads(batch_data)
        plt.rcParams.update({'font.size': 3, 'lines.linewidth': 2, 'lines.markersize': 1})
        with PdfPages(dump_file) as pdf:
            l = len(json_data.keys())
            for i, sim_key in enumerate(json_data.keys()):
                print("processing", i+1, "of", l)
                sim = json_data[sim_key]
                sim_type = sim['Batch type name']
                # the params that change per use case are defined in simulation/src/metrics/code/metrics_general_utils/fuzz_batch_requirements.py
                # the jsonObject fields names are defined in simulation/src/simulation/db/db_columns.py
                if sim_type == "SWARM":
                    sim_title = sim_type
                elif sim_type == "STOP_AND_GO":
                    ego_initial_velocity = sim["Scenario Params"]["global_params"]["general"]["initial_velocity"]
                    player_acceleration = sim["Scenario Params"]["player_params"]["stop_n_go"]["a"]
                    sim_title = sim_type + "_" + str(round(ego_initial_velocity, 1)) + "_" + str(round(player_acceleration, 1))
                elif sim_type == "APPROACH_SLOW_TARGET":
                    ego_initial_velocity = sim["Scenario Params"]["global_params"]["general"]["initial_velocity"]
                    player_velocity = sim["Scenario Params"]["player_params"]["slow_player"]["v2"]
                    sim_title = sim_type + "_" + str(round(ego_initial_velocity, 1)) + "_" + str(round(player_velocity, 1))
                elif sim_type == "LANE_SPLITS_NO_ACTORS" or sim_type == "SPEED_LIMIT_CHANGE":
                    track_pos_pair = sim["Scenario Params"]["global_params"]["Start_and_stop_Track_Positions"]["track_pos_pair"]
                    sim_title = sim_type + "_" + track_pos_pair
                elif sim_type == "LANE_SPLITS_ACTOR_FOLLOWING":
                    ego_initial_velocity = sim["Scenario Params"]["global_params"]["general"]["initial_velocity"]
                    track_pos_pair = sim["Scenario Params"]["global_params"]["Start_and_stop_Track_Positions"]["track_pos_pair"]
                    sim_title = sim_type + "_" + track_pos_pair + "_" + str(round(ego_initial_velocity, 1))
                elif sim_type == "LEAD_ACCEL" or sim_type == "LEAD_DECEL":
                    ego_initial_velocity = sim["Scenario Params"]["global_params"]["general"]["initial_velocity"]
                    player_velocity = sim["Scenario Params"]["player_params"]["slow_player"]["v2"]
                    player_acceleration = sim["Scenario Params"]["player_params"]["stop_n_go"]["a"]
                    sim_title = sim_type + "_" + str(round(ego_initial_velocity, 1)) + + "_" + str(round(player_acceleration, 1)) + "_" + str(round(player_velocity, 1))
                elif sim_type == "STOP_SIGN_ACTOR_DRIVE_THROUGH":
                    override_headway = sim["Scenario Params"]["player_params"]["lane_keeping"]["override_headway"]
                    sim_title = sim_type + "_" + str(round(override_headway, 1))
                elif sim_type == "STOP_SIGN_WITH_STOPPED_ACTOR":
                    rel_dis = sim["Scenario Params"]["player_params"]["Stand_and_start_parameters"]["rel_dis"]
                    sim_title = sim_type + "_" + str(round(rel_dis))
                elif sim_type == "STOP_SIGN_NO_ACTORS":
                    # this case has no varying params
                    sim_title = sim_type
                else:
                    sim_title = sim_type + "_" + sim_key
                log_file_name = path + '/../simulation/ground_truth_data/data/sim_' + sim_key + '/UC_logs/AV_Log_dm_main.log'
                visualize_logs.plot_dynamics(log_file_name)
                plt.suptitle(sim_title)
                pdf.savefig()
                plt.close()
            print("results were written to", dump_file)
