import numpy as np
import pandas as pd
import DataStore
import Client
import candidate
import node
import sys
import pickle

ALG_OPT   = 1
ALG_PGM   = 2
ALG_CHEAP = 3
ALG_ALL   = 4
ALG_KNAP  = 5
ALG_POT   = 6
NUM_OF_ALGS = 6

# client action: updated according to what client does
# 0: init, 1: hit upon access of DSs, 2: miss upon access of DSs, 3: high DSs cost, prefer beta, 4: no pos ind, pay beta
"""
key is an integer
"""

class Simulator(object):

    # init a list of empty DSs
    def init_DS_list(self):
        return [DataStore.DataStore(ID = i, size = self.DS_size, miss_rate_window = self.mr_win, bf_size = self.BF_size) for i in range(self.num_of_DSs)]
            
    def init_client_list(self):
        return [Client.Client(ID = i) for i in range(self.num_of_clients)]
    
    def __init__(self, alg_mode, DS_insert_mode, req_df, client_DS_dist, client_DS_BW, bw_regularization, beta, k_loc, DS_size = 1000, mr_win = 100, BF_size = 8000, alpha=0.5, rand_seed = 42, verbose = 0):
        """
        Return a Simulator object with the following attributes:
            alg_mode:           mode of client: defined by macros above
            DS_insert_mode:     mode of DS insertion (1: fix, 2: distributed, 3: ego)
            req_list:           array of keys of requests. each key is a string
            client_DS_dist:     2D array of hop-count distance to datastores. entry (i,j) is the distance from client i to DS j
            client_DS_BW:       2D array of bottleneck bandwidth to datastores. entry (i,j) is the BW from client i to DS j
            bw_regularization:  bandwidth regularization factor
            beta:               miss penalty
            k_loc:              number of DSs a missed key is inserted to
            DS_size:            size of DS (default 1000)
            mr_win:             miss_rate_window for miss rate update in each DS (default 100)
            BF_size:            size of Bloom filter to use in each DS (default 8000, compatible with default DS size of 1000, and FP ratio of 0.02)
            alpha:              weight for convex combination of dist-bw for calculating costs (default 0.5)
        """
        self.alg_mode = alg_mode
        self.DS_insert_mode = DS_insert_mode
        self.client_DS_dist = client_DS_dist
        self.client_DS_BW = client_DS_BW
        self.bw_reg = bw_regularization
        self.beta = beta
        self.k_loc = k_loc
        self.DS_size = DS_size
        self.mr_win = mr_win
        self.BF_size = BF_size
        self.alpha = alpha
        self.rand_seed = rand_seed

        self.num_of_clients = client_DS_dist.shape[0]
        self.num_of_DSs = client_DS_dist.shape[1]

        # client_DS_cost(i,j) will hold the access cost for client i accessing DS j
        self.client_DS_cost = 1 + self.alpha * self.client_DS_dist + (1 - self.alpha) * (self.bw_reg / self.client_DS_BW)


        self.DS_list = self.init_DS_list() #DS_list is the list of DSs
        self.mr_of_DS = np.zeros(self.num_of_DSs)
        
        self.req_df = req_df
        
        self.pos_DS_list = {}
        
        self.client_list = self.init_client_list()
        
        self.cur_req_cnt = float(-1)
        self.cur_pos_DS_list = [] #list of the DSs with pos' ind' for the current request

        self.total_cost = float(0)
        self.total_access_cost = float(0)
        self.access_cnt = float(0)
        self.hit_cnt = float(0)
        self.high_cost_mp_cnt = float(0) # counts the misses for cases where accessing DSs was too costly, so the alg' decided to access directly the mem
        self.non_comp_miss_cnt = float(0)
        self.comp_miss_cnt = float(0)
        self.num_DS_accessed = float(0)
        self.avg_DS_accessed_per_req = float(0)
        self.avg_DS_hit_ratio = float(0)

    # Returns an np.array of the DSs with positive ind'
    def get_pos_DS_list(self, key):
        return np.array([DS.ID for DS in self.DS_list if (key in DS.bf)])

    # Returns true iff key is found in at least of one of DSs specified by DS_index_list
    def req_in_DS_list(self, key, DS_index_list):
        for i in DS_index_list:
            if (key in self.DS_list[i]):
                return True
        return False

    def gather_statistics(self):
        self.total_cost = np.sum( [client.total_cost for client in self.client_list ] ) # $$$ in Gabi's code, it was normalized by beta
        self.total_access_cost = np.sum( [client.total_access_cost for client in self.client_list ] ) # $$$ in Gabi's code, it was normalized by beta
        self.access_cnt = np.sum( [client.access_cnt for client in self.client_list ] )
        self.hit_cnt = np.sum( [client.hit_cnt for client in self.client_list ] )
        self.hit_ratio = float(self.hit_cnt) / self.access_cnt
        self.high_cost_mp_cnt = np.sum( [client.high_cost_mp_cnt for client in self.client_list ] )
        self.non_comp_miss_cnt = np.sum( [client.non_comp_miss_cnt for client in self.client_list ] )
        self.comp_miss_cnt = np.sum( [client.comp_miss_cnt for client in self.client_list ] )
        self.num_DS_accessed = np.sum( [sum(client.num_DS_accessed) for client in self.client_list ] )
        self.avg_DS_accessed_per_req = float(self.num_DS_accessed) / self.access_cnt
        self.avg_DS_hit_ratio = np.average([float(DS.hit_cnt)/DS.access_cnt for DS in self.DS_list])

    def start_simulator(self):
        # print ('alg_mode=%d, kloc = %d, beta = %d, insertion_mode=%d' % (self.alg_mode, self.k_loc, self.beta, self.DS_insert_mode))
        np.random.seed(self.rand_seed)
        for req_ind in range(self.req_df.shape[0]):
            if self.DS_insert_mode == 3: # ego mode
                # re-assign the request to client 0 w.p. 1/(self.num_of_clients+1) and handle it. otherwise, put it in a random DS
                if np.random.rand() < 1/(self.num_of_clients+1):
                    self.req_df.set_value(req_ind, 'client_id', 0)
                    self.handle_request(self.req_df.iloc[req_ind])
                else:
                    self.insert_key_to_random_DSs(self.req_df.iloc[req_ind])
            else: # fix or distributed mode
                self.handle_request(self.req_df.iloc[req_ind])
                if self.DS_insert_mode == 2: # distributed mode
                    self.insert_key_to_closest_DS(self.req_df.iloc[req_ind])
        self.gather_statistics()
        print ('tot_cost=%.2f, tot_access_cost= %.2f, hit_ratio = %.2f, high_cost_mp_cnt = %d, non_comp_miss_cnt = %d, comp_miss_cnt = %d, access_cnt = %d' % (self.total_cost, self.total_access_cost, self.hit_ratio, self.high_cost_mp_cnt, self.non_comp_miss_cnt, self.comp_miss_cnt, self.access_cnt)        )
        
    def update_mr_of_DS(self):
        self.mr_of_DS = np.array([DS.mr_cur[-1] for DS in self.DS_list])

    def handle_request(self, req):
        self.cur_req_cnt += 1
        self.cur_pos_DS_list = self.get_pos_DS_list(req.key)
        self.update_mr_of_DS()

        # check if there are no positive indicators. in such a case this is a compulsory miss
        if self.cur_pos_DS_list.size == 0:
            self.client_list[req.client_id].comp_miss_cnt += 1
            self.client_list[req.client_id].total_cost += self.beta
            self.insert_key_to_DSs(req)

        # in all other cases, there are positive indicators
        elif self.alg_mode == ALG_CHEAP:
            self.access_cheapest(req.client_id, req)
        elif self.alg_mode == ALG_ALL:
            self.access_all(req.client_id, req)
        elif self.alg_mode == ALG_KNAP:
            self.access_knapsack(req.client_id, req)
        elif self.alg_mode == ALG_OPT:
            self.access_opt(req.client_id, req)
        elif self.alg_mode == ALG_POT:
            self.access_potential(req.client_id, req)
        elif self.alg_mode == ALG_PGM:
            self.access_pgm(req.client_id, req)
        else: 
            print ('Wrong alg_code')

    def insert_key_to_closest_DS(self, req):
        # check to see if one needs to insert key to closest cache too
        if self.DS_insert_mode == 2:
            self.DS_list[req.client_id].insert(req.key)

    def insert_key_to_random_DSs(self, req):
        # use the first location as the random DS to insert to.
        self.DS_list[req['0']].insert(req.key)

    def insert_key_to_DSs(self, req):
        # insert key to all k_loc DSs
        for i in range(self.k_loc):
            self.DS_list[req['%d'%i]].insert(req.key)
            
    def access_cheapest(self, client_id, req):

        # in case several DSs have minimum access, pick a random one among them.

        # pos_DS_list_costs will hold the list of costs of DSs with positive indications: in particular,
        # pos_DS_list_costs(i) will hold the access cost of the i-th DS in the list pos_DS_list_costs
        pos_DS_list_costs = np.take( self.client_DS_cost[client_id] , self.cur_pos_DS_list )

        # min_cost_pos_DS_indices will hold the the cheapest DSs with pos' ind'
        min_cost_pos_DS_indices = np.where( pos_DS_list_costs == np.min(pos_DS_list_costs))
        min_cost_pos_DS_list = np.take (self.cur_pos_DS_list , min_cost_pos_DS_indices)[0]
        access_DS_id = min_cost_pos_DS_list[np.random.randint(min_cost_pos_DS_list.size)]

        # check to see if it's too expensive to access the cheapest DS
        if self.client_DS_cost[client_id][access_DS_id] > self.beta:
            self.client_list[client_id].high_cost_mp_cnt += 1
            self.client_list[client_id].total_cost += self.beta
#           self.client_list[client_id].action[req.req_id] = 3
            self.insert_key_to_DSs(req)
        else:
            # update variables
            self.client_list[client_id].total_cost += self.client_DS_cost[client_id][access_DS_id]
            self.client_list[client_id].total_access_cost += self.client_DS_cost[client_id][access_DS_id]
            self.client_list[client_id].add_DS_accessed(req.req_id, [access_DS_id])
            self.client_list[client_id].access_cnt += 1

            # perform access. returns True if successful, and False otherwise
            if (self.DS_list[access_DS_id].access(req.key)): #hit
                self.client_list[client_id].hit_cnt += 1

            # Miss
            else:
                self.client_list[client_id].total_cost += self.beta
                # check if this is was a non compulsory miss
                if self.req_in_DS_list(req.key, self.cur_pos_DS_list):
                    self.client_list[client_id].non_comp_miss_cnt += 1
                else: # in this case it's a compulsory miss
                    self.client_list[client_id].comp_miss_cnt += 1
                self.insert_key_to_DSs(req)
        return

    def access_all(self, client_id, req, consider_beta = True):

        # check to see if it's too expensive to access the all of cur_pos_DS_list
        if consider_beta & (np.sum( np.take( self.client_DS_cost[client_id] , self.cur_pos_DS_list ) ) > self.beta):
            self.client_list[client_id].high_cost_mp_cnt += 1
            self.client_list[client_id].total_cost += self.beta
            self.insert_key_to_DSs(req)
        else:
            # update variables
            self.client_list[client_id].total_cost += np.sum( np.take( self.client_DS_cost[client_id] , self.cur_pos_DS_list ) )
            self.client_list[client_id].total_access_cost += np.sum( np.take( self.client_DS_cost[client_id] , self.cur_pos_DS_list ) )
            self.client_list[client_id].add_DS_accessed(req.req_id, self.cur_pos_DS_list)
            self.client_list[client_id].access_cnt += 1

            # perform access. returns True if successful, and False otherwise
            accesses = np.array([self.DS_list[DS_id].access(req.key) for DS_id in self.cur_pos_DS_list])
            if any(accesses):
                self.client_list[client_id].hit_cnt += 1

            # otherwise, the DS access is unsuccessful (and a compulsory miss, since we accessed all)
            else:
                self.client_list[client_id].total_cost += self.beta
                self.client_list[client_id].comp_miss_cnt += 1
                self.insert_key_to_DSs(req)
#               self.client_list[client_id].action[req.req_id] = 2
        return

    def access_opt(self, client_id, req):

        # get the list of datastores holding the request
        true_answer_DS_list = np.array([DS_id for DS_id in self.cur_pos_DS_list if (req.key in self.DS_list[DS_id])])
        # check to see if it is a compulsory miss
        if true_answer_DS_list.size == 0:
            self.client_list[client_id].total_cost += self.beta
            self.client_list[client_id].comp_miss_cnt += 1
#           self.client_list[client_id].action[req.req_id] = 2
        # otherwise, it is a hit for opt
        else:
            # find the cheapest DS holding th request
            access_DS_id = true_answer_DS_list[np.argmin( np.take( self.client_DS_cost[client_id] , true_answer_DS_list ) )]
            # check to see if it's too expensive to access the cheapest DS
            if self.client_DS_cost[client_id][access_DS_id] > self.beta:
                self.client_list[client_id].high_cost_mp_cnt += 1
                self.client_list[client_id].total_cost += self.beta
                self.insert_key_to_DSs(req)
#                self.client_list[client_id].action[req.req_id] = 3
            else:
                # update variables
                self.client_list[client_id].total_cost += self.client_DS_cost[client_id][access_DS_id]
                self.client_list[client_id].total_access_cost += self.client_DS_cost[client_id][access_DS_id]
                self.client_list[client_id].add_DS_accessed(req.req_id, [access_DS_id])
                self.client_list[client_id].access_cnt += 1
                # perform access. we know it will be successful
                self.DS_list[access_DS_id].access(req.key)
                self.client_list[client_id].hit_cnt += 1

        return
                
    def phi_cost(self, client_id, DS_index_list):
        return np.sum( np.take( self.client_DS_cost[client_id] , DS_index_list ) ) + self.beta * np.product( np.take( self.mr_of_DS , DS_index_list ) )
        
    def access_knapsack(self, client_id, req):
        
        client_log_weights = - np.log2([DS.mr_cur[-1] for DS in self.DS_list])
        client_weight_cost_ratios = np.multiply(client_log_weights , 1/self.client_DS_cost[client_id])
        
        singleton_DS_list_cost = np.array([self.phi_cost(client_id , np.array([DS_id])) for DS_id in range(self.num_of_DSs)])
        min_DS_singleton = self.cur_pos_DS_list[np.argmin( np.take( singleton_DS_list_cost , self.cur_pos_DS_list ) )]
        
        # df: dataframe holding the input for the client_id, and cur_pos_ind_DS_list:
        #   df[0]: DS_id, df[1]: (client,DS) cost, df[2]: (client,DS) weight cost ratio, df[3] (client,DS) log weights
        # we first calculate all the data for all DSs, and then just slice the DSs in cur_pos_DS_list
        df = pd.DataFrame(np.transpose([ range(self.num_of_DSs), self.client_DS_cost[client_id], client_weight_cost_ratios, client_log_weights ]))
        df = df.loc[df[0].isin(self.cur_pos_DS_list)]
        df = df.sort_values([2], ascending = False)
        unique_costs = np.unique(np.take( self.client_DS_cost[client_id] , self.cur_pos_DS_list))
        candidate_DS_cost = np.infty * np.ones((unique_costs.size, self.cur_pos_DS_list.size))
        for u, j_u in enumerate(unique_costs):
            df_u = df.loc[ df[1] <= j_u ]
            for t in range(df_u.shape[0]):
                candidate_DS_cost[u][t] = self.phi_cost(client_id , np.array(df.loc[ df[1] <= j_u ][0][:t+1]).astype('int'))
        min_DS_list_idx = np.unravel_index(candidate_DS_cost.argmin(), candidate_DS_cost.shape)
        min_DS_list = np.array(df[ df[1] <= unique_costs[min_DS_list_idx[0]] ][0][:min_DS_list_idx[1]+1], dtype='int')
        
        if singleton_DS_list_cost[min_DS_singleton] < candidate_DS_cost[min_DS_list_idx]:
            access_DS_list = np.array([min_DS_singleton])
        else:
            access_DS_list = np.sort(min_DS_list)
        
        # check to see if it's too expensive to access the access_DS_list
        if np.sum( np.take( self.client_DS_cost[client_id] , access_DS_list ) ) > self.beta:
            self.client_list[client_id].high_cost_mp_cnt += 1
            self.client_list[client_id].total_cost += self.beta
            self.insert_key_to_DSs(req)
#            self.client_list[client_id].action[req.req_id] = 3
        else:
            # update variables
            self.client_list[client_id].total_cost += np.sum( np.take( self.client_DS_cost[client_id] , access_DS_list ) )
            self.client_list[client_id].total_access_cost += np.sum( np.take( self.client_DS_cost[client_id] , access_DS_list ) )
            self.client_list[client_id].add_DS_accessed(req.req_id, access_DS_list)
            self.client_list[client_id].access_cnt += 1
            # perform access. returns True if successful, and False otherwise
            accesses = np.array([self.DS_list[DS_id].access(req.key) for DS_id in access_DS_list])
            if any(accesses): #hit
                self.client_list[client_id].hit_cnt += 1

            # otherwise, the DS access is unsuccessful (miss)
            else:
                self.client_list[client_id].total_cost += self.beta
                # check if this is was a non compulsory miss
                if self.req_in_DS_list(req.key, self.cur_pos_DS_list):
                    self.client_list[client_id].non_comp_miss_cnt += 1
                else: # in this case it's a compulsory miss
                    self.client_list[client_id].comp_miss_cnt += 1
                self.insert_key_to_DSs(req)
        return
    
    def access_potential(self, client_id, req):

        sorted_pos_DS_cost_indices = sorted(self.cur_pos_DS_list, key=self.client_DS_cost[client_id].__getitem__)
        sorted_pos_DS_mr_indices = sorted(self.cur_pos_DS_list, key=self.mr_of_DS.__getitem__)
        potential_values = [np.sum(np.take(self.client_DS_cost[client_id] , sorted_pos_DS_cost_indices[:k+1])) +
                            self.beta * np.product(np.take(self.mr_of_DS , sorted_pos_DS_mr_indices[:k+1]))
                            for k in range(self.cur_pos_DS_list.size)]
        k_min = np.argmin(potential_values)
        access_DS_list = np.sort(sorted_pos_DS_mr_indices[: k_min + 1])
        
        # check to see if it's too expensive to access the access_DS_list
        if np.sum( np.take( self.client_DS_cost[client_id] , access_DS_list ) ) > self.beta:
            self.client_list[client_id].high_cost_mp_cnt += 1
            self.client_list[client_id].total_cost += self.beta
            self.insert_key_to_DSs(req)
        else:
            # update variables
            self.client_list[client_id].total_cost += np.sum( np.take( self.client_DS_cost[client_id] , access_DS_list ) )
            self.client_list[client_id].total_access_cost += np.sum( np.take( self.client_DS_cost[client_id] , access_DS_list ) )
            self.client_list[client_id].add_DS_accessed(req.req_id, access_DS_list)
            self.client_list[client_id].access_cnt += 1

            # perform access. the function access() returns True if successful, and False otherwise
            accesses = np.array([self.DS_list[DS_id].access(req.key) for DS_id in access_DS_list])
            if any(accesses): # Hit
                self.client_list[client_id].hit_cnt += 1

            # Miss
            else:
                self.client_list[client_id].total_cost += self.beta
                # check if this is was a non compulsory miss
                if self.req_in_DS_list(req.key, self.cur_pos_DS_list):
                    self.client_list[client_id].non_comp_miss_cnt += 1
                else: # in this case it's a compulsory miss
                    self.client_list[client_id].comp_miss_cnt += 1
                self.insert_key_to_DSs(req)
        return

    def access_pgm(self, client_id, req):

        self.cur_pos_DS_list = [int(i) for i in self.cur_pos_DS_list] # cast cur_pos_DS_list to int

        # Partition stage
        ###############################################################################################################
        # leaf_of_DS (i,j) will hold the leaf to which DS with cost (i,j) belongs, that is, log_2 (DS(i,j))
        self.leaf_of_DS = np.array(np.floor(np.log2(self.client_DS_cost)))
        self.leaf_of_DS = self.leaf_of_DS.astype('uint8')

        # leaves_of_DSs_w_pos_ind will hold the leaves of the DSs with pos' ind'
        cur_num_of_leaves = np.max (np.take(self.leaf_of_DS[client_id], self.cur_pos_DS_list)) + 1

        # DSs_in_leaf[j] will hold the list of DSs which belong leaf j, that is, the IDs of all the DSs with access in [2^j, 2^{j+1})
        DSs_in_leaf = [[]]
        for leaf_num in range (cur_num_of_leaves):
            DSs_in_leaf.append ([])
        for ds in (self.cur_pos_DS_list):
            DSs_in_leaf[self.leaf_of_DS[client_id][ds]].append(ds)

        # Generate stage
        ###############################################################################################################
        # leaf[j] will hold the list of candidate DSs of V^0_j in the binary tree
        leaf = [[]]
        for leaf_num in range (cur_num_of_leaves-1): # Append additional cur_num_of_leaves-1 leaves
            leaf.append ([])

        for leaf_num in range (cur_num_of_leaves):

            # df_of_DSs_in_cur_leaf will hold the IDs, miss rates and access costs of the DSs in the current leaf
            num_of_DSs_in_cur_leaf = len(DSs_in_leaf[leaf_num])
            df_of_DSs_in_cur_leaf = pd.DataFrame({
                'DS ID': DSs_in_leaf[leaf_num],
                'mr': np.take(self.mr_of_DS, DSs_in_leaf[leaf_num]), #miss rate
                'ac': np.take(self.client_DS_cost[client_id], DSs_in_leaf[leaf_num]) #access cost
            })


            df_of_DSs_in_cur_leaf.sort_values(by=['mr'], inplace=True) # sort the DSs in non-dec. order of miss rate


            leaf[leaf_num].append(candidate.candidate ([], 1, 0)) # Insert the empty set to the leaf
            cur_mr = 1
            cur_ac = 0

            # For each prefix_len \in {1 ... number of DSs in the current leaf},
            # insert the prefix at this prefix_len to the current leaf
            for pref_len in range (1, num_of_DSs_in_cur_leaf+1):
                cur_mr *= df_of_DSs_in_cur_leaf.iloc[pref_len - 1]['mr']
                cur_ac += df_of_DSs_in_cur_leaf.iloc[pref_len - 1]['ac']
                leaf[leaf_num].append(candidate.candidate(df_of_DSs_in_cur_leaf.iloc[range(pref_len)]['DS ID'], cur_mr, cur_ac))

        # Merge stage
        ###############################################################################################################
        r = np.ceil(np.log2(self.beta)).astype('uint8')
        num_of_lvls = (np.ceil(np.log2 (cur_num_of_leaves))).astype('uint8') + 1
        if (num_of_lvls == 1): # Only 1 leaf --> nothing to merge. The candidate full solutions will be merely those in this single leaf
            cur_lvl_node = leaf
        else:
            prev_lvl_nodes = leaf
            num_of_nodes_in_prev_lvl = cur_num_of_leaves
            num_of_nodes_in_cur_lvl = np.ceil (cur_num_of_leaves / 2).astype('uint8')
            for lvl in range (1, num_of_lvls):
                cur_lvl_node = [None]*num_of_nodes_in_cur_lvl
                for j in range (num_of_nodes_in_cur_lvl):
                    if (2*(j+1) > num_of_nodes_in_prev_lvl): # handle edge case, when the merge tree isn't a full binary tree
                        cur_lvl_node[j] = prev_lvl_nodes[2*j]
                    else:
                        # print ('req_id = ', req.req_id, '\n')
                        cur_lvl_node[j] = node.merge(prev_lvl_nodes[2*j], prev_lvl_nodes[2*j+1], r, self.beta)
                num_of_nodes_in_prev_lvl = num_of_nodes_in_cur_lvl
                num_of_nodes_in_cur_lvl = (np.ceil(num_of_nodes_in_cur_lvl / 2)).astype('uint8')
                prev_lvl_nodes = cur_lvl_node

        min_final_candidate_phi = self.beta + 1 # Will hold the total cost among by all final sols checked so far
        for final_candidate in cur_lvl_node[0]:  # for each of the candidate full solutions
            final_candidate_phi = final_candidate.phi(self.beta)
            if (final_candidate_phi < min_final_candidate_phi): # if this sol' is cheaper than any other sol' found so far', take this new sol'
                final_sol = final_candidate
                min_final_candidate_phi = final_candidate_phi

        # print ('final sol is ', access_DS_list, '. Phi of final sol is ', min_full_sol_phi)

        if (len(final_sol.DSs_IDs) == 0): # the alg' decided to not access any DS, although we know there's at least 1 pos' ind' (otherwise access_pgm isn't called)
            self.client_list[client_id].high_cost_mp_cnt += 1
            self.client_list[client_id].total_cost += self.beta
            self.insert_key_to_DSs(req)
            return

        # Add the costs and IDs of the selected DSs to the statistics
        self.client_list[client_id].total_cost        += final_sol.ac
        self.client_list[client_id].total_access_cost += final_sol.ac
        self.client_list[client_id].add_DS_accessed(req.req_id, final_sol.DSs_IDs)
        self.client_list[client_id].access_cnt += 1
        # Now we know that PGM decided to access at least 1 DS
        # perform access. the function access() returns True if successful, and False otherwise
        accesses = np.array([self.DS_list[DS_id].access(req.key) for DS_id in final_sol.DSs_IDs])
        if any(accesses): #hit
            self.client_list[client_id].hit_cnt += 1

        # Miss
        else:
            self.client_list[client_id].total_cost += self.beta
            if self.req_in_DS_list(req.key, self.cur_pos_DS_list): # Is it a non compulsory miss?
                self.client_list[client_id].non_comp_miss_cnt += 1 # yep --> inc. relevant cntr
            else:
                self.client_list[client_id].comp_miss_cnt += 1 # compulsory miss

            # insert the missed item to the DSs
            self.insert_key_to_DSs(req)
        return

