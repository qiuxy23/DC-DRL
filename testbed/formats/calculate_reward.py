import torch
import pandas as pd


def calculate_user_reward(path, ddl):

    table1 = pd.read_table(path, header=0, sep=' ', names=['id', 'local', 'edge', 'overall'])
    res = 0
    for overall in table1.overall:
        if overall > ddl:
            res += 1
    return res


fail = 0

for i in range(1, 7):
    fail += calculate_user_reward('client_' + str(i) + '.log', 600)

print(fail)