# # ===========================================================
# # Parse results for variable Palse Positive rate simulations
# # ===========================================================
#
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import datetime
import python_simulator as sim

# Currently need to specify the exact filename you would like to open and process
f = open ('../res/2019_09_04_00_47_41_pickle_sim_dict_DS_size_200_1600_beta_100_kloc_1_FP_0.02', 'rb')
full_sim_dict = pickle.load(f)
f.close()
DS_size = 1000

# def get_total_cost(full_sim_dict, alg_num):
#     return full_sim_dict[FP_rate][alg_num].total_cost / (full_sim_dict[FP_rate][alg_num].cur_req_cnt + 1)

def get_total_cost_ratio (full_sim_dict, FP_rate, alg_num):
    return full_sim_dict[FP_rate][alg_num].total_cost / full_sim_dict[FP_rate][sim.ALG_OPT].total_cost

FP_rate_vals = [0.01, 0.02, 0.03, 0.04]

# total_cost_ratio_dict = {sim.ALG_OPT: [], sim.ALG_PGM: [], sim.ALG_CHEAP: [], sim.ALG_ALL: [], sim.ALG_KNAP: [], sim.ALG_POT: []}

# for j, FP_rate in enumerate(FP_rate_vals):
#     for alg_num in [sim.ALG_OPT, sim.ALG_PGM, sim.ALG_CHEAP, sim.ALG_ALL, sim.ALG_KNAP, sim.ALG_POT]:
#         total_cost_ratio_dict[alg_num].append(float(get_total_cost(full_sim_dict,DS_size,alg_num)) / get_total_cost(full_sim_dict,DS_size,sim.ALG_OPT ))

alg_name = ['\opt', '\pgmalg', '\cpi', '\epi', '\\umb', '\pot']
# print out the results in the format required for tikz
for j, alg_num in enumerate([sim.ALG_OPT, sim.ALG_CHEAP, sim.ALG_ALL, sim.ALG_KNAP, sim.ALG_POT, sim.ALG_PGM]): # in the homogeneous setting, no need to run 3 (Knap) since it is equivalent to 6 (Pot)
	if (alg_num == sim.ALG_OPT):
		continue
	# % Code for printing data in a suitable format for tikz table
	print('{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} bb'.format(alg_name[alg_num-1], \
		get_total_cost_ratio(full_sim_dict, 0.01, alg_num), \
		get_total_cost_ratio(full_sim_dict, 0.02, alg_num), \
		get_total_cost_ratio(full_sim_dict, 0.03, alg_num), \
		get_total_cost_ratio(full_sim_dict, 0.04, alg_num)))


# % Code for printing data in a suitable format for tikz plot
# 	print('{} (0.01, {:.2f}) (0.02, {:.2f}) (0.03, {:.2f}) (0.04, {:.2f})'.format(alg_name[alg_num-1], \
# 		get_total_cost_ratio(full_sim_dict, 0.01, alg_num), \
# 		get_total_cost_ratio(full_sim_dict, 0.02, alg_num), \
# 		get_total_cost_ratio(full_sim_dict, 0.03, alg_num), \
# 		get_total_cost_ratio(full_sim_dict, 0.04, alg_num)))
