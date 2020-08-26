# ========================================================================
# Parse results for sim where DS size is fixed, while beta and k_loc vary
# ========================================================================

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import python_simulator as sim

def get_ac_to_opt_tc_ratio (sim_dict, kloc, beta, alg_num):
    return sim_dict[beta][kloc][alg_num].total_access_cost / sim_dict[beta][kloc][sim.ALG_OPT].total_cost

def get_tc_ratio (sim_dict, kloc, beta, alg_num):
    return sim_dict[beta][kloc][alg_num].total_cost / sim_dict[beta][kloc][sim.ALG_OPT].total_cost

filename='2019_09_03_20_47_23_sim_dict_beta_10000'
sim_dict = {}
f = open(filename, 'rb')
beta = int(filename.split ('_')[-1])
sim_dict[beta] = pickle.load(f)
f.close()
print ('beta = ', beta)

for alg_num, alg_name in enumerate (['\opt', '\pgmalg', '\cpi', '\epi', '\\umb', '\pot'], start=1):
	if alg_name not in (['\opt']):
		print('& {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} bb'.format(alg_name, \
			get_ac_to_opt_tc_ratio (sim_dict, 1, beta, alg_num), \
			get_tc_ratio (sim_dict, 1, beta, alg_num), \
			get_ac_to_opt_tc_ratio(sim_dict, 3, beta, alg_num), \
			get_tc_ratio(sim_dict, 3, beta, alg_num), \
			get_ac_to_opt_tc_ratio (sim_dict, 5, beta, alg_num), \
			get_tc_ratio (sim_dict, 5, beta, alg_num)))

#		print('& {} & {:.2f} & {:.2f} '.format(alg_name, \
#			get_ac_to_opt_tc_ratio (sim_dict, 3, beta, alg_num), \
#			get_tc_ratio (sim_dict, 3, beta, alg_num)))
