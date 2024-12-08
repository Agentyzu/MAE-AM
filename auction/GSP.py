import numpy as np


def GSP(b, pctr, pos_norm, q, constant, len_content, max_len):
    ecpm = b * pctr
    sigma = np.argsort(-ecpm)
    Prom = np.zeros(q)
    ecpm_k = np.zeros(q)

    for k in range(q):
        ecpm_k[k] = b[sigma[k]] * pctr[sigma[k]] * pos_norm[k]
        Prom[k] = 1 / q

    payments = np.zeros(q)
    for k in range(q - 1):
        payments[k] = ecpm[sigma[k + 1]]
    payments[q - 1] = 0

    utilities = np.zeros(q)
    SW = 0

    for k in range(q):
        utilities[k] = b[sigma[k]] * Prom[k] - payments[k]
        SW += b[sigma[k]] * pctr[sigma[k]] * pos_norm[k] * (np.log(Prom[k]) + constant)

    return sigma, Prom, payments, utilities, SW
