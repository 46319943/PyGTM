import numpy as np
from numba import prange, njit

# @njit(parallel=False)
@njit(parallel=True)
def bug_func():
    n_cols = 10
    n_rows = 1

    for _ in prange(10):
        matrix = np.zeros((n_rows, n_cols))
        vector = np.zeros((n_rows,))

        # This line cause the process terminated unexpectedly. This should be the bug of numba.
        vector_ = vector[:, np.newaxis]

        value = matrix - vector_

    return None


if __name__ == '__main__':
    bug_func()
