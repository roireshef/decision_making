import json

import matplotlib.pyplot as plt
from decision_making.paths import Paths
from decision_making.test.proving_ground import visualize_logs
from matplotlib.backends.backend_pdf import PdfPages


if __name__ == "__main__":
    path = Paths.get_repo_path()
    # Enter path of json file to analyze here:
    json_filename = path + '/../simulation/ground_truth_data/sim_metadata/run_info_file_2019_09_04_09_20_52_743122.json'
    with open(json_filename, 'r') as f:
        batch_data = f.read()
        batch_data = "{" + batch_data.replace("\n", "").replace("\'", "")[0:-1] + "}"
        dump_file = path + "/../logs/batch_" + json_filename.split("run_info_file_")[1].replace("json", "pdf")
        json_data = json.loads(batch_data)
        plt.rcParams.update({'font.size': 3, 'lines.linewidth': 2, 'lines.markersize': 1})
        with PdfPages(dump_file) as pdf:
            i = 1
            l = len(json_data.keys())
            for sim_key in json_data.keys():
                print("processing", i, "of", l)
                sim = json_data[sim_key]
                sim_type = sim['Batch type name']
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
                else:
                    sim_title = sim_type + "_" + sim_key
                log_file_name = path + '/../simulation/ground_truth_data/data/sim_' + sim_key + '/UC_logs/AV_Log_dm_main.log'
                visualize_logs.plot_dynamics(log_file_name)
                plt.suptitle(sim_title)
                pdf.savefig()
                plt.close()
                i += 1
                if i == 2:
                    break
            print("results were written to", dump_file)
