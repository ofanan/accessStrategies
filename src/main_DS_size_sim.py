import numpy as np
import pandas as pd
import python_simulator as sim
from tictoc import tic, toc
import datetime
import sys
import pickle

# A main file for running a sim of access strategies for different DSs (Data Stores) sizes.

num_of_DSs = 19
num_of_clients = 19

## This produces a random matrix with a specific value on the diagonal.
## Can be used to produce a random distance matrix (with 0 on diag) and a BW matrix (with infty on diag)
#def gen_rand_matrix(num, diag_value = 0, max_dist = 30):
#    a = np.tril(np.random.randint(1,max_dist,(num,num)), k=-1)
#    return a + np.transpose(a) + np.diag(diag_value * np.ones(num))

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
trace_df = pd.read_csv('C:/Users/ofanan/Google Drive/Comnet/BF_n_Hash/Python_Infocom19/trace_5m_%d.csv' % file_index)
#trace_df.info(memory_usage='deep')
trace_df = reduce_trace_mem_print(trace_df)
#trace_df.info(memory_usage='deep')

## For debugging / shorter runs, pick a prefix of the trace
## Eventually, req_df is the variable fed into the simulation.
## The default value for req_df is the whole trace, trace_df
trace_length=10000
req_df = trace_df.head(trace_length)
#req_df.info(memory_usage='deep')

## Code for generating the random dist and BW matrices
#client_DS_dist = gen_rand_matrix(17)
#client_DS_BW = gen_rand_matrix(17, diag_value = np.infty)

## Generate matrix for the fully homogeneous settings. This would evnetually result in client_DS_cost all 1
client_DS_dist = np.zeros((num_of_clients,num_of_DSs)) # np.ones((num_of_clients,num_of_DSs)) - np.eye(17)
client_DS_BW = np.ones((num_of_clients,num_of_DSs)) # + np.diag(np.infty * np.ones(num_of_clients))
bw_regularization = 0. # np.max(np.tril(client_DS_BW,-1))

# Sizes of the bloom filters (number of cntrs), for chosen cache sizes, k=5 hash funcs, and designed false positive rate.
# The values are taken from https://hur.st/bloomfilter
# online resource calculating the optimal values
BF_size_for_DS_size = {}
BF_size_for_DS_size[0.01] = {20: 197, 40: 394, 60: 591, 80: 788, 100: 985, 200: 1970, 400: 3940, 600: 5910, 800: 7880, 1000: 9849, 1200: 11819, 1400: 13789, 1600: 15759, 2000: 19698, 2500: 24623, 3000: 29547}
BF_size_for_DS_size[0.02] = {20: 164, 40: 328, 60: 491, 80: 655, 100: 819, 200: 1637, 400: 3273, 600: 4909, 800: 6545, 1000: 8181, 1200: 9817, 1400: 11453, 1600: 13089}
BF_size_for_DS_size[0.03] = {1000: 7299}
BF_size_for_DS_size[0.04] = {1000: 6711}

# Loop over all data store sizes, and all algorithms, and collect the data
def run_sim_collection(DS_size_vals, FP_rate, beta, k_loc, req_df, client_DS_dist, client_DS_BW, bw_regularization):
    DS_insert_mode = 1

    main_sim_dict = {}
    for DS_size in DS_size_vals:
        BF_size = BF_size_for_DS_size[FP_rate][DS_size]
        print ('DS_size = ', DS_size)
        DS_size_sim_dict = {}
        for alg_mode in [sim.ALG_OPT]: #, sim.ALG_ALL, sim.ALG_CHEAP, sim.ALG_POT, sim.ALG_PGM]: # in the homogeneous setting, no need to run Knap since it is equivalent to 6 (Pot)
            tic()
            print (datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
            sm = sim.Simulator(alg_mode, DS_insert_mode, req_df, client_DS_dist, client_DS_BW, bw_regularization, beta, k_loc, DS_size = DS_size, BF_size = BF_size)
            sm.start_simulator()
            toc()
            DS_size_sim_dict[alg_mode] = sm
        main_sim_dict[DS_size] = DS_size_sim_dict
    return main_sim_dict

## Choose parameters for running simulator    
beta = 100
FP_rate = 0.02
k_loc = 3

DS_size_vals = [200] #, 400, 600, 800, 1000, 1200, 1400, 1600]
main_sim_dict = run_sim_collection(DS_size_vals, FP_rate, beta, k_loc, req_df, client_DS_dist, client_DS_BW, bw_regularization)

time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

sys.setrecursionlimit(50000)
res_file = open('res/%s_pickle_sim_dict_DS_size_%d_%d_beta_%d_kloc_%d_FP_%.2f' % (time_str ,DS_size_vals[0], DS_size_vals[-1], beta, k_loc, FP_rate) , 'wb')
pickle.dump(main_sim_dict , res_file)
res_file.close()

