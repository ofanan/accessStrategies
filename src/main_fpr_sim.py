# ========================================================================
# Performs sims where DS size is fixed, while beta and k_loc vary
# ========================================================================
import numpy as np
import pandas as pd
from tictoc import tic, toc
import datetime
import sys
import pickle

import python_simulator as sim
from gen_requests import gen_requests
from gen_requests import optimal_BF_size_per_cache_size

num_of_DSs = 19
num_of_clients = 19

## Generate the requests to be fed into the simulation. For debugging / shorter runs, pick a prefix of the trace, of length max_trace_length
max_trace_length=10000
requests = gen_requests ('C:/Users/ofanan/Documents/traces/wiki/wiki1.1190448987.csv', max_trace_length)

# load the OVH network distances and BWs
client_DS_dist_df = pd.read_csv('../../Python_Infocom19/ovh_dist.csv',index_col=0)
client_DS_dist = np.array(client_DS_dist_df)
client_DS_BW_df = pd.read_csv('../../Python_Infocom19/ovh_bw.csv',index_col=0)
client_DS_BW = np.array(client_DS_BW_df)
bw_regularization = np.max(np.tril(client_DS_BW,-1))

# Sizes of the bloom filters (number of cntrs), for chosen cache sizes, k=5 hash funcs, and designed false positive rate.
BF_size_for_DS_size = optimal_BF_size_per_cache_size ()

def run_sim_collection(DS_size, FP_rate_vals, beta, k_loc, requests, client_DS_dist, client_DS_BW, bw_regularization):
    DS_insert_mode = 1

    main_sim_dict = {}
    for FP_rate in FP_rate_vals:
        print ('FP_rate = ', FP_rate)
        BF_size = BF_size_for_DS_size[FP_rate][DS_size]
        DS_size_sim_dict = {}
        for alg_mode in [sim.ALG_OPT, sim.ALG_PGM, sim.ALG_CHEAP, sim.ALG_ALL, sim.ALG_KNAP, sim.ALG_POT]:
            tic()
            print (datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
            sm = sim.Simulator(alg_mode, DS_insert_mode, requests, client_DS_dist, client_DS_BW, bw_regularization, beta, k_loc, DS_size = DS_size, BF_size = BF_size)
            sm.start_simulator()
            toc()
            DS_size_sim_dict[alg_mode] = sm
        main_sim_dict[FP_rate] = DS_size_sim_dict
    return main_sim_dict

## Choose parameters for running simulator
beta = 100
k_loc = 5
DS_size = 1000
FP_rate_vals = [0.01, 0.02, 0.03, 0.04]
main_sim_dict = run_sim_collection(DS_size, FP_rate_vals, beta, k_loc, requests, client_DS_dist, client_DS_BW, bw_regularization)

time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

sys.setrecursionlimit(50000)
res_file = open('%s_sim_dict_DS_size_%d_FP_rate_%.2f_%.2f_beta_%d_kloc_%d' % (time_str ,DS_size, FP_rate_vals[0], FP_rate_vals[-1], beta, k_loc) , 'wb')
pickle.dump(main_sim_dict , res_file)
res_file.close()

