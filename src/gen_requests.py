import pandas
## This reduces the memory print of the trace by using the smallest type that still supports the values in the trace
## Note: this configuration can support up to 2^8 locations, and traces of length up to 2^32
def reduce_trace_mem_print(trace_df):
    new_trace_df = trace_df
    new_trace_df['req_id'] = trace_df['req_id'].astype('uint32')        # id (running cnt number) of the request
    new_trace_df['key'] = trace_df['key'].astype('uint32')              # key. No need for value in the sim'
    new_trace_df['client_id'] = trace_df['client_id'].astype('uint8')   # client to which this request is assigned
    for i in range(17):
        new_trace_df['%d'%i] = trace_df['%d'%i].astype('uint8')
    return new_trace_df


# Sizes of the bloom filters (number of cntrs), for chosen cache sizes, k=5 hash funcs, and designed false positive rate.
# The values are taken from https://hur.st/bloomfilter
# online resource calculating the optimal values
def optimal_BF_size_per_cache_size ():
    BF_size_for_DS_size = {}
    BF_size_for_DS_size[0.01] = {20: 197, 40: 394, 60: 591, 80: 788, 100: 985, 200: 1970, 400: 3940, 600: 5910, 800: 7880, 1000: 9849, 1200: 11819, 1400: 13789, 1600: 15759, 2000: 19698, 2500: 24623, 3000: 29547}
    BF_size_for_DS_size[0.02] = {20: 164, 40: 328, 60: 491, 80: 655, 100: 819, 200: 1637, 400: 3273, 600: 4909, 800: 6545, 1000: 8181, 1200: 9817, 1400: 11453, 1600: 13089}
    BF_size_for_DS_size[0.03] = {1000: 7299}
    BF_size_for_DS_size[0.04] = {1000: 6711}
    BF_size_for_DS_size[0.05] = {20: 126, 40: 251, 60: 377, 80: 502, 100: 628, 200: 1255, 400: 2510, 600: 3765, 800: 5020, 1000: 6275, 1200: 7530, 1400: 8784, 1600: 10039}
    return BF_size_for_DS_size 


def gen_requests (trace_file_name, numOfReq):
    trace_df = reduce_trace_mem_print(pandas.read_csv(trace_file_name))
    return trace_df.head (numOfReq)

