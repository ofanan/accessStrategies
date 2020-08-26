import numpy as np
import pandas as pd
import datetime as dt
import hashlib
import matplotlib as plt
import datetime

file_index = 6
filename = 'currenttmp_%d.txt' % file_index

df = pd.read_csv(filename, sep=' ', header=None)

num_of_clients = 19
num_of_locations = 19

# take only "read" requests
df = df.loc[df[3] == '-']

# sort trace by time, breaking ties by sequence number
df[1] = pd.to_datetime(df[1], unit='s')
df = df.sort_values([1,0])
df = df.reset_index(drop=True)

# take only first 5 minutes
df['time_diff'] = (df[1]-df.iloc[0][1]) # calculate time difference
#time_delta_5min = datetime.timedelta(0, 300, 0) # 300s == 5min
#under_5m = (df['time_diff'] < time_delta_5min) # boolean array of entries satisfying condition
#df_5m = df[under_5m].copy()
time_delta_2min = datetime.timedelta(0, 300, 0) # 120s == 5min
under_Xm = (df['time_diff'] < time_delta_2min) # boolean array of entries satisfying condition
df_Xm = df[under_Xm].copy()

# generate hashes of URLs. name column 'key'
unique_urls = np.unique(df_Xm[2])
url_lut_dict = dict(zip(unique_urls , range(unique_urls.size))) # generate dictionary to serve as a LUT of unique_key -> integer
#url_lut_df = pd.DataFrame(url_lut_dict.items())
hashed_url_array = np.array([url_lut_dict[url] for url in df_Xm[2]]).astype('uint32')

# generate client assignments for each request
client_assignment = np.random.RandomState(seed=42).randint(0 , num_of_clients , df_Xm.shape[0]).astype('uint8')

# generate request ids
req_id = np.array(range(df_Xm.shape[0])).astype('uint32')

# generate permutation for each unique key in trace
unique_permutations_array  = np.array([np.random.RandomState(seed=i).permutation(range(num_of_locations)) for i in range(unique_urls.size)]).astype('uint8')
permutation_lut_dict = dict(zip(unique_urls , unique_permutations_array)) # generate dictionary to serve as a LUT of unique_key -> permutation
# generate full permutations array for all requests. identical keys will get identical permutations
permutations_array = np.array([permutation_lut_dict[url] for url in df_Xm[2]]).astype('uint8')
permutations_df = pd.DataFrame(permutations_array)

trace_df = pd.DataFrame(np.transpose([req_id, hashed_url_array, client_assignment]))
trace_df.columns = ['req_id', 'key', 'client_id']

full_trace_df = pd.concat([ trace_df, permutations_df ], axis=1)

full_trace_df.to_csv('trace_5m_%d.csv' % file_index, index=False, header=True)

## check memory space used by dataframe
#trace_df.info(memory_usage='deep')

## validate string values in dataframe
#non_string_keys = [i for i, key in enumerate(df[2]) if not isinstance(key,basestring)]
#for i, key in enumerate(df[2].head()):
#    if isinstance(key,basestring):
#        print i

## historgram
#count_df = trace.value_counts()
#count_df[count_df > 1].plot()
#hist_array = np.zeros(int(np.log2(count_df.max())) + 1)
#for i in range(int(np.log2(count_df.max())) + 1):
#    hist_array[i] = sum(count_df[(count_df >= 2**i) & (count_df < 2**(i+1))])
