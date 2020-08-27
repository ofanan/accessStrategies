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

num_of_DSs = 19
num_of_clients = 19

## Generate the requests to be fed into the simulation. For debugging / shorter runs, pick a prefix of the trace, of length max_trace_length
max_trace_length=10000
requests = gen_requests ('C:/Users/ofanan/Google Drive/Comnet/BF_n_Hash/Python_Infocom19/trace_5m_6.csv', max_trace_length)

# load the OVH network distances and BWs
client_DS_dist_df = pd.read_csv('../../Python_Infocom19/ovh_dist.csv',index_col=0)
client_DS_dist = np.array(client_DS_dist_df)
client_DS_BW_df = pd.read_csv('../../Python_Infocom19/ovh_bw.csv',index_col=0)
client_DS_BW = np.array(client_DS_BW_df)
bw_regularization = np.max(np.tril(client_DS_BW,-1))

# taken from https://hur.st/bloomfilter
# online resource calculating the optimal values
# these values are for k=5 hash functions, and a target of false positive = 0.01, 0.02, 0.05}

BF_size_for_DS_size = {}
BF_size_for_DS_size[0.01] = {20: 197, 40: 394, 60: 591, 80: 788, 100: 985, 200: 1970, 400: 3940, 600: 5910, 800: 7880, 1000: 9849, 1200: 11819, 1400: 13789, 1600: 15759, 2000: 19698, 2500: 24623, 3000: 29547}
BF_size_for_DS_size[0.02] = {20: 164, 40: 328, 60: 491, 80: 655, 100: 819, 200: 1637, 400: 3273, 600: 4909, 800: 6545, 1000: 8181, 1200: 9817, 1400: 11453, 1600: 13089}
BF_size_for_DS_size[0.05] = {20: 126, 40: 251, 60: 377, 80: 502, 100: 628, 200: 1255, 400: 2510, 600: 3765, 800: 5020, 1000: 6275, 1200: 7530, 1400: 8784, 1600: 10039}
DS_size = 1000
FP_rate = 0.02
BF_size = BF_size_for_DS_size[FP_rate][DS_size]

beta = 10000
def run_sim_collection(DS_size, BF_size, beta, req_df, client_DS_dist, client_DS_BW, bw_regularization):
    DS_insert_mode = 1

    main_sim_dict = {}
    for k_loc in [1]: #, 3, 5]:
        print ('k_loc = ', k_loc)
        k_loc_sim_dict = {}
        for alg_mode in [sim.ALG_OPT]: #, sim.ALG_PGM, sim.ALG_CHEAP, sim.ALG_ALL, sim.ALG_KNAP, sim.ALG_POT]:
            tic()
            sm = sim.Simulator(alg_mode, DS_insert_mode, req_df, client_DS_dist, client_DS_BW, bw_regularization, beta, k_loc, DS_size = DS_size, BF_size = BF_size)
            sm.start_simulator()
            toc()
            k_loc_sim_dict[alg_mode] = sm
        main_sim_dict[k_loc] = k_loc_sim_dict
    return main_sim_dict
    
main_sim_dict = run_sim_collection(DS_size, BF_size, beta, req_df, client_DS_dist, client_DS_BW, bw_regularization)

time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

sys.setrecursionlimit(50000)
res_file = open('%s_sim_dict_beta_%d' % (time_str , beta) , 'wb')
pickle.dump(main_sim_dict, res_file)
res_file.close()

