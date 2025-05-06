import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import permutations

# 定義一個小型成本矩陣（3個tracker 對應 3個detection）
C = np.array([
    [4, 1, 3],
    [2, 0, 5],
    [3, 2, 2]
])

# 匈牙利演算法解
row_ind, col_ind = linear_sum_assignment(C)
hungarian_total_cost = C[row_ind, col_ind].sum()
hungarian_matching = list(zip(row_ind, col_ind))

# 暴力解法：窮舉所有匹配方式
n = C.shape[0]
min_cost = float('inf')
best_permutation = None

for perm in permutations(range(n)):
    cost = sum(C[i, perm[i]] for i in range(n))
    if cost < min_cost:
        min_cost = cost
        best_permutation = list(enumerate(perm))

best_permutation_cost = min_cost

print(hungarian_matching, hungarian_total_cost, best_permutation, best_permutation_cost)