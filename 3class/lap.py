import numpy as np
from scipy.optimize import linear_sum_assignment

__version__ = "0.5.14"


def lapjv(cost, extend_cost=True, cost_limit=np.inf, return_cost=True):
    cost = np.asarray(cost, dtype=float)
    n_rows, n_cols = cost.shape
    x = -np.ones(n_rows, dtype=int)
    y = -np.ones(n_cols, dtype=int)
    if cost.size == 0:
        return (0.0, x, y) if return_cost else (x, y)
    rows, cols = linear_sum_assignment(cost)
    keep = cost[rows, cols] <= cost_limit
    rows, cols = rows[keep], cols[keep]
    x[rows] = cols
    y[cols] = rows
    return (float(cost[rows, cols].sum()), x, y) if return_cost else (x, y)
