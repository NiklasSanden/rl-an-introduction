import numpy as np
from queue import PriorityQueue
from queue import Empty

def argmax(array):
    '''
    The numpy argmax function breaks ties by choosing the first occurence. This implementation breaks ties uniformly at random.
    '''
    assert (len(array.shape) == 1), 'argmax expects a 1d array'
    assert (array.size > 0),        'argmax expects non-empty array'

    indices = []
    best = float('-inf')
    for i in range(len(array)):
        if array[i] > best:
            best = array[i]
            indices = [i]
        elif array[i] == best:
            indices.append(i)
    return np.random.choice(indices)

class PrioritizedSweepingQueue(object):
    '''
    Uses queue.PriorityQueue to sort the state and action pairs by a priority. It is sorted in ascending order
    so negate the priority if you want it to give the largest priority first.
    If a state-action pair already exists, only the one with the best priority will end up in the queue.
    '''
    def __init__(self):
        self.q = PriorityQueue()
        self.entries = dict()
        self.tie_breaker = 0

    def put(self, state, action, priority):
        key = (state, action)
        if key in self.entries:
            old_priority = self.entries[key][0]
            if old_priority <= priority:
                return
            self._delete(key)
        entry = [priority, self.tie_breaker, key]
        self.tie_breaker += 1
        self.q.put_nowait(entry)
        self.entries[key] = entry
    
    def _delete(self, key):
        entry = self.entries.pop(key)
        entry[-1] = None
    
    def pop(self):
        while self.q.qsize() > 0:
            _, _, key = self.q.get_nowait()
            if key:
                del self.entries[key]
                return key
        raise Empty('Tried to pop from an empty queue')

    def empty(self):
        return not self.entries
