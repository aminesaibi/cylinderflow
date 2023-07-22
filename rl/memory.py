import random

class Memory:
    """ Rotating memory buffer """
    
    def __init__(self, capacity):
        """
        capacity : maximum number of elements.
        buffer : data container.
        ptr : index pointer to the position that receives the data.
        size : number of elements currently present in the container.
        """
        
        self.capacity = capacity
        self.buffer = [None]*capacity
        self.ptr = 0
        self.size = 0
        self.batch_size = 64
    
    def append(self, elmt):
        self.buffer[self.ptr] = elmt
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity) 
    
    def append_list(self, lst):
        for elmt in lst:
            self.append(elmt)

    def sample(self, sample_size):
        if sample_size <= self.size :
            sample_indices = random.sample(range(self.size), sample_size)
            sample = [self.buffer[i] for i in sample_indices]
            return sample
        else:
            raise Exception("sample size must be smaller than buffer size !")
