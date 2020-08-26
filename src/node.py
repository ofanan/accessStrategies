import numpy as np
import numpy as np
import pandas as pd
import candidate

# Merges 2 input nodes.
# Inputs: 2 nodes to merge, H and L, and r = log(beta)
# Ouput: res_node - the result node, res_node.
def merge(H, L, r, beta):

    # init res_node, to an empty set, and dummy candidates with miss rate = 1.1, s.t. every real candidate at the same log' range will replace them
    res_node = [None]*(r+1)
    res_node[0] = candidate.candidate([], 1, 0) # Init the result node to include only the empty set
    for t in range(1, r + 1):
        res_node[t] = candidate.candidate ([], 1.1, 0)
    for A in H:
        for B in L:
            new_cand_mr = A.mr * B.mr
            new_cand_ac = A.ac + B.ac
            if (new_cand_ac == 0): # if both candidates are empty set, there's nothing to merge
                continue
            if (new_cand_ac > beta): # if cost of the merged node will be > beta, it's useless to create it
                continue
            t = (np.floor(np.log2(new_cand_ac))).astype(int) + 1
            if (res_node[t].mr > new_cand_mr): # if the mr of new cand < current minimal mr at the same bucket - take the new cand'
                res_node[t] = candidate.candidate (list(set(A.DSs_IDs) | set (B.DSs_IDs)), new_cand_mr, new_cand_ac)

    # remove all the dummy candidates, with mr=1.1
    for t in range(r, 0, -1): #remove top-down, as while removing, res_node may shrink, and not be of size r+1 anymore
        if (res_node[t].mr == 1.1):
            res_node.pop (t)
    return res_node

