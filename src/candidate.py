# A candidate sol' for the DSS prob', consists of a list of DSs, theirs aggregate (mult' of) miss rate, and aggregate (sum of) accs cost
class candidate (object):
    def __init__(self, DSs, mr, ac): # Gen a new candidate, given a list of DSs it contains, theirs total mr and ac (accs cost)
        self.DSs_IDs = [int(i) for i in DSs]
        self.mr    = mr
        self.ac    = ac

    def phi (self, beta):
        return self.ac + beta * self.mr
        