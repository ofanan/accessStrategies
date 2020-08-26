import numpy as np

class Request(object):
    
    def __init__(self, ID, client_id, key):
        self.ID = ID
        self.client_id = client_id
        self.key = key

        
class Client(object):
    
    def __init__(self, ID):
        """
        Return a Client object with the following attributes:
            ID:                 client ID 
        """
        self.ID = ID
        
        self.total_cost = 0
        self.total_access_cost = 0
        self.req_cnt = 0
        self.access_cnt = 0
        self.hit_cnt = 0
        self.high_cost_mp_cnt = 0
        self.non_comp_miss_cnt = 0
        self.comp_miss_cnt = 0

        # dictionary describing for every req_id of client: 0: init, 1: hit upon access of DSs, 2: miss upon access of DSs, 3: high DSs cost, prefer beta, 4: no pos ind, pay beta
        self.action 			= {}

        self.cur_mr_list 		= {} # dictionary containing the mr list req_id of client
        self.DS_accessed 		= {} # dictionary containing DSs accessed for every req_id of client where access takes place
        self.num_DS_accessed 	= []

    def add_DS_accessed(self, req_id, DS_index_list):
        self.DS_accessed[req_id] = DS_index_list
        self.num_DS_accessed.append(len(DS_index_list))
        
