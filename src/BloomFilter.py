import numpy as np
import mmh3


class BloomFilter(object):
    
    def __init__(self, size = 8000, hash_count = 5, max_array_val = 255):
        """
        Returns a CountingBloomFilter object with the following attributes:
            size:           the number of counters for mapping keys (default 1000)
            hash_count:     number of hash functions to use for mapping each key (default 5)
            max_array_val:  maximum value each counter can hold (default 255)
        """
        self.size = size
        self.hash_count = hash_count
        self.array = np.zeros(size, dtype='uint')
        self.max_array_val = max_array_val
 
    def add(self, key):
        """
        adds a key to the Bloom filter
        updates all the hashes corresponding to the new key: increases each by 1 (no more than allowed maximum)
        """
        for seed in range(self.hash_count):
            entry = mmh3.hash(key, seed) % self.size
            if self.array[entry] < self.max_array_val:
                self.array[entry] += 1

    def remove(self, key):
        """
        removes a key from the Bloom filter
        updates all the hashes corresponding to the key: decreases each by 1 (no less than 0)
        """
        for seed in range(self.hash_count):
            entry = mmh3.hash(key, seed) % self.size
            if self.array[entry] > 0:
                self.array[entry] -= 1

    def __contains__(self, key):
        """
        checks if all hash functions corresponding to key are strictly positive
        """
        for seed in range(self.hash_count):
            entry = mmh3.hash(key, seed) % self.size
            if self.array[entry] == 0:
                return False
        return True

    def lookup(self, key):
        """
        checks if all hash functions corresponding to key are strictly positive
        """
        for seed in range(self.hash_count):
            entry = mmh3.hash(key, seed) % self.size
            if self.array[entry] == 0:
                return False
        return True
        
    def get_fpr(self):
        """
        return the estimate false-positive ratio
        CHECK THE FORMULA!!!
        """
        return (sum(self.array > 0) / self.size) ** self.hash_count
