# ========================================================================
# Parse results for sim where DS size is fixed, while beta and k_loc vary
# ========================================================================

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import python_simulator as sim

def get_ncmp_ratio(sim_dict, kloc, beta, alg_num):
    return beta*float(sim_dict[beta][kloc][alg_num].high_cost_mp_cnt + sim_dict[beta][kloc][alg_num].non_comp_miss_cnt) / sim_dict[beta][kloc][sim.ALG_OPT].total_access_cost

def get_ac_ratio (sim_dict, kloc, beta, alg_num):
    return sim_dict[beta][kloc][alg_num].total_access_cost / sim_dict[beta][kloc][sim.ALG_OPT].total_access_cost

def get_tc_ratio (sim_dict, kloc, beta, alg_num):
    return sim_dict[beta][kloc][alg_num].total_cost / sim_dict[beta][kloc][sim.ALG_OPT].total_cost

filename= '2019_08_01_06_41_12_sim_dict_beta_1000' #filename should be in the format 'date_dict_beta_X' where beta = X
sim_dict = {}
f = open(filename, 'rb')
beta = int(filename.split ('_')[-1])
sim_dict[beta] = pickle.load(f)
f.close()
kloc = 1
if (sim_dict[beta][kloc][sim.ALG_OPT].DS_size != 1000):
    print ('wrong DS_size')
    exit (1)
print('cpi: ac = {:.2f} & tc = {:.2f} '.format(get_ac_ratio(sim_dict, kloc, beta, sim.ALG_CHEAP), get_tc_ratio(sim_dict, kloc, beta, sim.ALG_CHEAP)))
print('pgm: ac = {:.2f} & tc = {:.2f} '.format(get_ac_ratio(sim_dict, kloc, beta, sim.ALG_PGM), get_tc_ratio(sim_dict, kloc, beta, sim.ALG_PGM)))
