import random

def random_exit():
    '''
    Implementation of random exit
    '''
    spatial = random.random()
    temporal = random.random()
    return spatial > 0.5, temporal > 0.5