# ========================================================================
# Performs sims where DS size is fixed, while beta and k_loc vary
# ========================================================================
import numpy as np
import pandas as pd
import python_simulator as sim
from tictoc import tic, toc
import datetime
import sys
import pickle

num_of_DSs = 19
num_of_clients = 19

## This reduces the memory print of the trace by using the smallest type that still supports the values in the trace
## Note: this configuration can support up to 2^8 locations, and traces of length up to 2^32
def reduce_trace_mem_print(trace_df):
    new_trace_df = trace_df
    new_trace_df['req_id'] = trace_df['req_id'].astype('uint32')
    new_trace_df['key'] = trace_df['key'].astype('uint32')
    new_trace_df['client_id'] = trace_df['client_id'].astype('uint8')
    for i in range(17):
        new_trace_df['%d'%i] = trace_df['%d'%i].astype('uint8')
    return new_trace_df

## Which file index to use. Relevant to the case where we want to use multiple trace files
file_index = 6

## Read the trace, and reduce its memory print
trace_df = pd.read_csv('../../Python_Infocom19/trace_5m_%d.csv' % file_index)
trace_df = reduce_trace_mem_print(trace_df)

## For debugging / shorter runs, pick a prefix of the trace
## Eventually, req_df is the variable fed into the simulation.
## The default value for req_df is the whole trace, trace_df
#trace_length=1000
req_df = trace_df#.head(trace_length)

# load the OVH network distances and BWs
client_DS_dist_df = pd.read_csv('../../Python_Infocom19/ovh_dist.csv',index_col=0)
client_DS_dist = np.array(client_DS_dist_df)
client_DS_BW_df = pd.read_csv('../../Python_Infocom19/ovh_bw.csv',index_col=0)
client_DS_BW = np.array(client_DS_BW_df)
bw_regularization = np.max(np.tril(client_DS_BW,-1))

# taken from https://hur.st/bloomfilter
# online resource calculating the optimal values
# these values are for k=5 hash functions, and a target of FP in {0.01, 0.02, 0.05}
BF_size_for_DS_size = {}
BF_size_for_DS_size[0.01] = {20: 197, 40: 394, 60: 591, 80: 788, 100: 985, 200: 1970, 400: 3940, 600: 5910, 800: 7880, 1000: 9849, 1200: 11819, 1400: 13789, 1600: 15759, 2000: 19698, 2500: 24623, 3000: 29547}
BF_size_for_DS_size[0.02] = {20: 164, 40: 328, 60: 491, 80: 655, 100: 819, 200: 1637, 400: 3273, 600: 4909, 800: 6545, 1000: 8181, 1200: 9817, 1400: 11453, 1600: 13089}
BF_size_for_DS_size[0.03] = {1000: 7299}
BF_size_for_DS_size[0.04] = {1000: 6711}

def run_sim_collection(DS_size, FP_rate_vals, beta, k_loc, req_df, client_DS_dist, client_DS_BW, bw_regularization):
    DS_insert_mode = 1

    main_sim_dict = {}
    for FP_rate in FP_rate_vals:
        print ('FP_rate = ', FP_rate)
        BF_size = BF_size_for_DS_size[FP_rate][DS_size]
        DS_size_sim_dict = {}
        for alg_mode in [sim.ALG_OPT, sim.ALG_PGM, sim.ALG_CHEAP, sim.ALG_ALL, sim.ALG_KNAP, sim.ALG_POT]:
            tic()
            print (datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
            sm = sim.Simulator(alg_mode, DS_insert_mode, req_df, client_DS_dist, client_DS_BW, bw_regularization, beta, k_loc, DS_size = DS_size, BF_size = BF_size)
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
main_sim_dict = run_sim_collection(DS_size, FP_rate_vals, beta, k_loc, req_df, client_DS_dist, client_DS_BW, bw_regularization)

time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

sys.setrecursionlimit(50000)
res_file = open('%s_sim_dict_DS_size_%d_FP_rate_%.2f_%.2f_beta_%d_kloc_%d' % (time_str ,DS_size, FP_rate_vals[0], FP_rate_vals[-1], beta, k_loc) , 'wb')
pickle.dump(main_sim_dict , res_file)
res_file.close()

