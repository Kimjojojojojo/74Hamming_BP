import numpy as np
def tanh(x):
    return np.tanh(x)
def atanh(x):
    return np.arctanh(x)

y = [1, 0, 1, 0, 1, 0, 1]
e = 0.1
N = len(y)
H = np.array([[1, 0, 1, 0, 1, 0, 1],
              [0, 1, 1, 0, 0, 1, 1],
              [0, 0, 0, 1, 1, 1, 1]])
row = np.size(H, 0)
col = np.size(H, 1)
print(row, col)

py_x = np.zeros((N, 2))
for n in range(N):
    if y[n] == 0:
        py_x[n, 0] = 1 - e
        py_x[n, 1] = e
    if y[n] == 1:
        py_x[n, 0] = e
        py_x[n, 1] = 1 - e

LLRc = np.zeros(N)
for n in range(N):
    LLRc[n] = np.log(py_x[n, 1]) - np.log(py_x[n, 0])

LLR_mu = np.zeros((row, col))
LLR_mu_hat = np.zeros((col, row))
max_iter = 100
# initial mu update
for r in range(row):
    for c in range(col):
        if H[r][c] == 1:
            LLR_mu[r][c] = LLRc[c]
#
# print(LLR_mu)

mu_list = []
mu_hat_list = []
for m in range(max_iter):

    if m == 0: # initial mu
        mu_list.append(LLR_mu)
        continue

    mu_tmp = np.zeros((row, col))
    mu_hat_tmp = np.zeros((col, row))

    print(mu_list[m-1])
    # mu_hat update
    for r in range(row):
        for c in range(col):
            if H[r][c] == 1:
                # ---- tanh computation start ---- #
                tanh_tmp = 1
                for co in range(col):
                    if H[r][co] == 1:
                        tanh_tmp *= tanh(mu_list[m-1][r][co] / 2)
                # ---- tanh computation end ---- #
                mu_hat_tmp[c][r] = 2 * atanh(tanh_tmp)
    mu_hat_list.append(mu_hat_tmp)


    for r in range(row):
        for c in range(col):
            if H[r][c] == 1:
                mu_tmp[r][c] = mu_list[m-1][r][c] + sum(mu_hat_tmp[c]) - mu_hat_tmp[c][r]

    mu_list.append(mu_tmp)

output = np.zeros(N)
for n in range(N):
    output[n] = (sum(mu_hat_list[-1][n]) + LLRc[n]>1)
print(output)
