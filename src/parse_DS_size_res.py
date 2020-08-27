# # ===========================================================
# # results for variable DS_size simulations
# # ===========================================================
#
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import datetime
import python_simulator as sim

# Currently need to specify the exact filename you would like to open and process
#f = open('2019_09_05_02_08_23_pickle_sim_dict_DS_size_200_1600_beta_100_kloc_5_FP_0.02', 'rb')
f = open ('../res/2019_09_04_00_47_41_pickle_sim_dict_DS_size_200_1600_beta_100_kloc_1_FP_0.02', 'rb')
full_sim_dict = pickle.load(f)
f.close()

def get_total_cost(full_sim_dict, DS_size, alg_num):
    return full_sim_dict[DS_size][alg_num].total_cost / (full_sim_dict[DS_size][alg_num].cur_req_cnt + 1)

def get_hit_ratio(full_sim_dict, DS_size, alg_num):
    return full_sim_dict[DS_size][alg_num].hit_cnt / (full_sim_dict[DS_size][alg_num].cur_req_cnt + 1)

DS_size_vals = [200, 400, 600, 800, 1000, 1200, 1400, 1600]

total_cost_ratio_dict = {sim.ALG_OPT: [], sim.ALG_ALL: [], sim.ALG_CHEAP: [], sim.ALG_POT: [], sim.ALG_PGM: []}
total_cost_dict = {sim.ALG_OPT: [], sim.ALG_ALL: [], sim.ALG_CHEAP: [], sim.ALG_POT: [], sim.ALG_PGM: []}
hit_ratio_dict = {sim.ALG_OPT: [], sim.ALG_ALL: [], sim.ALG_CHEAP: [], sim.ALG_POT: [], sim.ALG_PGM: []}

for j, DS_size in enumerate(DS_size_vals):
    for alg_num in [sim.ALG_OPT, sim.ALG_ALL, sim.ALG_CHEAP, sim.ALG_POT, sim.ALG_PGM]:
        total_cost_ratio_dict[alg_num].append(float(get_total_cost(full_sim_dict,DS_size,alg_num)) / get_total_cost(full_sim_dict,DS_size,sim.ALG_OPT ))

alg_name = ['\opt', '\pgmalg', '\cpi', '\epi', '\\umb', '\pot']
# print out the results in the format required for tikz
for j, alg_num in enumerate([sim.ALG_CHEAP, sim.ALG_ALL,sim.ALG_POT,sim.ALG_PGM]): # in the homogeneous setting, no need to run 3 (Knap) since it is equivalent to 6 (Pot)
	print (alg_name[alg_num-1])
	zipped_data = zip(DS_size_vals, total_cost_ratio_dict[alg_num])
	zipped_data_as_list = list (zipped_data)
	print (zipped_data_as_list)
