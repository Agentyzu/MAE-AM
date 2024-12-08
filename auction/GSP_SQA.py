import numpy as np


def GSP_SQA(b, pctr, pos_norm, q, constant, len_content, max_len):
    ecpm = b * pctr
    sigma = np.argsort(-ecpm)
    Prom = np.zeros(q)
    ecpm_k = np.zeros(q)

    for k in range(q):
        ecpm_k[k] = b[sigma[k]] * pctr[sigma[k]] * pos_norm[k]
        if k == 0:
            Prom[k] = len_content[sigma[k]] / max_len
            if Prom[k] > 1:
                Prom[k] = 1
        else:
            Prom[k] = 0.01

    payments = np.zeros(q)
    for k in range(q):
        if k == 0:
            payments[k] = ecpm[sigma[k + 1]]
        else:
            payments[k] = 0

    utilities = np.zeros(q)
    SW = 0

    for k in range(q):
        utilities[k] = b[sigma[k]] * Prom[k] - payments[k]
        if Prom[k] != 0.01:
            SW += b[sigma[k]] * pctr[sigma[k]] * pos_norm[k] * (np.log(Prom[k]) + constant)


    return sigma, Prom, payments, utilities, SW
