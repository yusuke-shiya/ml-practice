import numpy as np


def approach_napier(n):
    return (1 + 1 / n) ** n


n_list = [2, 4, 10, 10000]  # このリストにさらに大きな数を追加する
for n in n_list:
    print("a_" + str(n) + " =", approach_napier(n))
