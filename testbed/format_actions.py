import torch
import numpy as np
import random

actions = np.load('save_actions_8/actions_-0.4301893366078192.npy')

for i_user in range(8):
    action = [random.randint(0, 1) for val in actions[:, i_user * 2]]
    # action = [min(1, int(val) + 1) for val in actions[:, i_user * 2]]
    np.save('actions_' + str(i_user + 1), action)
    print(action)

