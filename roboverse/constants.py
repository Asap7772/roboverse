import numpy as np
import os

def get_array():
    lo=np.array((0.8, .4, -0.1))
    hi=np.array((.4, -.2, -.34))

    path = 'test.npy'
    if not os.path.exists(path):
        print('recreated')
        with open(path, 'wb') as f:
            np.save(f, np.random.uniform(lo, hi, (100000,) + hi.shape))

    with open(path, 'rb') as f:
        rand_pos = np.load(f)
    return rand_pos