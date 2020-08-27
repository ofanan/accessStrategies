import numpy as np
import pandas as pd
from tictoc import tic, toc
import datetime
import sys
import pickle

import python_simulator as sim
from gen_requests import gen_requests
from gen_requests import optimal_BF_size_per_cache_size
# A main file for running a sim of access strategies for different DSs (Data Stores) sizes.

num_of_DSs = 19
num_of_clients = 19

## This produces a random matrix with a specific value on the diagonal.
## Can be used to produce a random distance matrix (with 0 on diag) and a BW matrix (with infty on diag)
#def gen_rand_matrix(num, diag_value = 0, max_dist = 30):
#    a = np.tril(np.random.randint(1,max_dist,(num,num)), k=-1)
#    return a + np.transpose(a) + np.diag(diag_value * np.ones(num))


## Generate the requests to be fed into the simulation. For debugging / shorter runs, pick a prefix of the trace, of length max_trace_length
max_trace_length=10000
requests = gen_requests ('C:/Users/ofanan/Documents/traces/wiki/wiki1.1190448987.csv', max_trace_length)

## Code for generating a random dist and BW matrices
#client_DS_dist = gen_rand_matrix(17)
#client_DS_BW = gen_rand_matrix(17, diag_value = np.infty)

## Generate matrix for the fully homogeneous settings. This would evnetually result in client_DS_cost all 1
client_DS_dist = np.zeros((num_of_clients,num_of_DSs)) # np.ones((num_of_clients,num_of_DSs)) - np.eye(17)
client_DS_BW = np.ones((num_of_clients,num_of_DSs)) # + np.diag(np.infty * np.ones(num_of_clients))
bw_regularization = 0. # np.max(np.tril(client_DS_BW,-1))

# Sizes of the bloom filters (number of cntrs), for chosen cache sizes, k=5 hash funcs, and designed false positive rate.
BF_size_for_DS_size = optimal_BF_size_per_cache_size ()

# Loop over all data store sizes, and all algorithms, and collect the data
def run_sim_collection(DS_size_vals, FP_rate, beta, k_loc, requests, client_DS_dist, client_DS_BW, bw_regularization):
    DS_insert_mode = 1

    main_sim_dict = {}
    for DS_size in DS_size_vals:
        BF_size = BF_size_for_DS_size[FP_rate][DS_size]
        print ('DS_size = ', DS_size)
        DS_size_sim_dict = {}
        for alg_mode in [sim.ALG_OPT]: #, sim.ALG_ALL, sim.ALG_CHEAP, sim.ALG_POT, sim.ALG_PGM]: # in the homogeneous setting, no need to run Knap since it is equivalent to 6 (Pot)
            tic()
            print (datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
            sm = sim.Simulator(alg_mode, DS_insert_mode, requests, client_DS_dist, client_DS_BW, bw_regularization, beta, k_loc, DS_size = DS_size, BF_size = BF_size)
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

main_sim_dict = run_sim_collection(DS_size_vals, FP_rate, beta, k_loc, requests, client_DS_dist, client_DS_BW, bw_regularization)

time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

sys.setrecursionlimit(50000)
res_file = open('../res/%s_pickle_sim_dict_DS_size_%d_%d_beta_%d_kloc_%d_FP_%.2f' % (time_str ,DS_size_vals[0], DS_size_vals[-1], beta, k_loc, FP_rate) , 'wb')
pickle.dump(main_sim_dict , res_file)
res_file.close()

