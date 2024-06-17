import numpy as np


def tanh(x):
    return np.tanh(x)


def atanh(x):
    return np.arctanh(x)
def ML_detector(y, e):
    N = len(y)
    # Convert string to list
    y_list = list(y)

    # Convert list back to string

    py_0 = 0
    py_1 = 0
    if type(y) == str:
        for n in range(N):
            if y_list[n] == '0':  # Use y_list instead of y
                py_0 = 1 - e
                py_1 = e
            if y_list[n] == '1':  # Use y_list instead of y
                py_0 = e
                py_1 = 1 - e

            if py_0 > py_1:
                y_list[n] = '0'  # Modify y_list instead of y
            if py_0 < py_1:
                y_list[n] = '1'  # Modify y_list instead of y

        y_modified = ''.join(y_list)
        return y_modified

    for n in range(N):
        if y_list[n] == 0:  # Use y_list instead of y
            py_0 = 1 - e
            py_1 = e
        if y_list[n] == 1:  # Use y_list instead of y
            py_0 = e
            py_1 = 1 - e

        if py_0 > py_1:
            y_list[n] = 0  # Modify y_list instead of y
        if py_0 < py_1:
            y_list[n] = 1  # Modify y_list instead of y
    return y_list


def BP_detector(y, codewords, e):
    N = len(y)

    py_x = np.zeros((7, 2))
    for n in range(N):
        if y[n] == 0:
            py_x[n, 0] = 1 - e
            py_x[n, 1] = e
        if y[n] == 1:
            py_x[n, 0] = e
            py_x[n, 1] = 1 - e


    # 0
    p0 = np.zeros(N)
    for n in range(N):
        codewords_0 = []
        for idx, codeword in enumerate(codewords):
            # print(type(codeword[n]))
            if codeword[n] == '0':
                #print(codeword)
                codewords_0.append(codeword)
        #print(codewords_0)
        sum_tmp = 0
        for code_0 in codewords_0:
            multiple_tmp = 1
            for i in range(N):
                if i == n:
                    continue

                if y[i] == code_0[i]:
                    multiple_tmp = multiple_tmp * (1 - e)
                else:
                    multiple_tmp = multiple_tmp * e

            sum_tmp = sum_tmp + multiple_tmp

        if y[n] == '0':
            p0[n] = sum_tmp * (1 - e)
        else:
            p0[n] = sum_tmp * e

    # 1
    p1 = np.zeros(N)
    for n in range(N):
        codewords_1 = []
        for idx, codeword in enumerate(codewords):
            # print(type(codeword[n]))
            if codeword[n] == '1':
                # print(codeword)
                codewords_1.append(codeword)
        #print(codewords_1)
        sum_tmp = 0
        for code_1 in codewords_1:
            multiple_tmp = 1
            for i in range(N):
                if i == n:
                    continue

                if y[i] == code_1[i]:
                    multiple_tmp = multiple_tmp * (1 - e)
                else:
                    multiple_tmp = multiple_tmp * e

            sum_tmp = sum_tmp + multiple_tmp

        if y[n] == '1':
            p1[n] = sum_tmp * (1 - e)
        else:
            p1[n] = sum_tmp * e

    detected = np.zeros(N)
    for n in range(N):
        if p0[n] > p1[n]:
            detected[n] = 0
        if p0[n] < p1[n]:
            detected[n] = 1
        if p0[n] == p1[n]:
            detected[n] = y[n]

    detected_str = ''.join(map(str, detected.astype(int)))

    return detected_str

def LBP_detector(y, codewords, e):
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

        if m == 0:  # initial mu
            mu_list.append(LLR_mu)
            continue

        mu_tmp = np.zeros((row, col))
        mu_hat_tmp = np.zeros((col, row))

        print(mu_list[m - 1])
        # mu_hat update
        for r in range(row):
            for c in range(col):
                if H[r][c] == 1:
                    # ---- tanh computation start ---- #
                    tanh_tmp = 1
                    for co in range(col):
                        if H[r][co] == 1:
                            tanh_tmp *= tanh(mu_list[m - 1][r][co] / 2)
                    # ---- tanh computation end ---- #
                    mu_hat_tmp[c][r] = 2 * atanh(tanh_tmp)
        mu_hat_list.append(mu_hat_tmp)

        for r in range(row):
            for c in range(col):
                if H[r][c] == 1:
                    mu_tmp[r][c] = mu_list[m - 1][r][c] + sum(mu_hat_tmp[c]) - mu_hat_tmp[c][r]

        mu_list.append(mu_tmp)

    output = np.zeros(N)
    for n in range(N):
        output[n] = (sum(mu_hat_list[-1][n]) + LLRc[n] > 1)


    return output

LLR = LBP_detector([1, 0, 1, 0, 1, 0, 1],1 ,0.1)
print(LLR)

    # print(p0)
    # print(p1)
