import numpy as np
import pandas as pd
import python_simulator as sim
import sys
import pickle
import matplotlib.pyplot as plt

# Load some of OVH data
client_DS_dist_df = pd.read_csv('../../Python_Infocom19/ovh_dist.csv',index_col=0)
client_DS_dist = np.array(client_DS_dist_df)
client_DS_BW_df = pd.read_csv('../../Python_Infocom19/ovh_bw.csv',index_col=0)
client_DS_BW = np.array(client_DS_BW_df)
bw_regularization = np.max(np.tril(client_DS_BW,-1))

# Some dummy params' values, just for generating the simulator, which will calculate the costs
DS_size = 1000
FP_rate = 0.02
BF_size = 8000
beta = 100
alg_mode = sim.ALG_OPT
DS_insert_mode = 1
file_index = 6
k_loc = 1
trace_df = pd.read_csv('../../Python_Infocom19/trace_5m_%d.csv' % file_index)
req_df = trace_df.head(100)
sm = sim.Simulator(alg_mode, DS_insert_mode, req_df, client_DS_dist, client_DS_BW, bw_regularization, beta, k_loc,
                   DS_size=DS_size, BF_size=BF_size)
cost_array = sm.client_DS_cost

# # build histogram data
all_unique_costs = cost_array[np.nonzero(cost_array)]
print (all_unique_costs)
max_val = np.ceil(np.max(all_unique_costs)).astype('int')
hist_values = np.histogram(all_unique_costs, bins=max_val, range=(1,max_val+1))[0] / float(all_unique_costs.size)
plt.bar(range(1,30), hist_values)

for i in range(29):
    if (hist_values[i]>0):
        print ('(%d, %.2f)' % (i+1,hist_values[i]))
    
# for i in range (10): #(np.max(sim_dict_beta[1][3].client_list[0].num_DS_accessed)):
#     print ('rg')
    # print gamad
    # print ('=', i+1, ':', len([j for j, x in enumerate(sim_dict_beta[1][6].client_list[0].num_DS_accessed) if (x > i) and (x <= i+1)])
