import numpy as np
import copy

def Prom_DSIC(b, pctr, pos_norm, q, constant, len_content, max_len, beta, s):
    ecpm = b * pctr
    sigma = np.argsort(-ecpm)
    Prom = np.zeros(q)
    ecpm_k = np.zeros(q)

    b_hat = copy.deepcopy(b)
    b_hat[sigma[s]] = b[sigma[s]] * beta

    for k in range(q):
        ecpm_k[k] = b_hat[sigma[k]] * pctr[sigma[k]] * pos_norm[k]

    for k in range(q):
        Prom[k] = ecpm_k[k] / np.sum(ecpm_k)

    payments = np.zeros(q)
    if q == 1:
        payments[0] = ecpm[sigma[1]]
    else:
        for k in range(q):
            w_minus_i = np.sum(ecpm_k) - ecpm_k[k]
            t = ecpm_k[k] / w_minus_i
            payments[k] = (w_minus_i / (pctr[sigma[k]] * pos_norm[k])) * (np.log(1 + t) - t / (1 + t))

    utilities = np.zeros(q)
    for k in range(q):
        utilities[k] = b[sigma[k]] * Prom[k] - payments[k]

    SW = np.sum(ecpm_k * (np.log(ecpm_k / np.sum(ecpm_k)) + constant))

    return sigma, Prom, payments, utilities, SW
