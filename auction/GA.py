
import numpy as np


def GA(b, pctr, pos_norm, q, constant, len_content, max_len):
    ecpm = b * pctr
    sigma = np.argsort(-ecpm)
    Prom = np.zeros(q)
    ecpm_k = np.zeros(q)

    for k in range(q):
        ecpm_k[k] = b[sigma[k]] * pctr[sigma[k]] * pos_norm[k]

    remaining_len = max_len
    for k in range(q):
        idx = sigma[k]
        if remaining_len > 0:
            ad_len = len_content[idx]
            alloc_len = min(ad_len, remaining_len)
            Prom[k] = alloc_len / max_len
            remaining_len -= alloc_len
        else:
            Prom[k] = 0.01

    num_allocated_ads = np.sum(Prom > 0.01)

    payments = np.zeros(num_allocated_ads)

    # GSP 支付公式
    # for k in range(num_allocated_ads - 1):
    #     payments[k] = ecpm[sigma[k + 1]]
    # payments[num_allocated_ads - 1] = 0

    if num_allocated_ads == 1:
        payments[0] = ecpm[sigma[1]]
    else:
        for k in range(num_allocated_ads):
            w_minus_i = np.sum(ecpm_k) - ecpm_k[k]
            t = ecpm_k[k] / w_minus_i
            payments[k] = (w_minus_i / (pctr[sigma[k]] * pos_norm[k])) * (np.log(1 + t) - t / (1 + t))

    utilities = np.zeros(num_allocated_ads)
    SW = 0

    for k in range(num_allocated_ads):
        utilities[k] = b[sigma[k]] * Prom[k] - payments[k]

    for k in range(q):
        if Prom[k] != 0.01:
            SW += b[sigma[k]] * pctr[sigma[k]] * pos_norm[k] * (np.log(Prom[k]) + constant)

    return sigma, Prom, payments, utilities, SW


# constant = 3
# max_len = 150
# b = np.random.uniform(0, 1, 10)
# pos_norm = [0.9 ** (k+1) for k in range(10)]
# pctr = np.random.uniform(0.3, 0.5, 10)
# len_content = [60 for _ in range(10)]
# for q in range(2, 8):
#     sigma, Prom, payments, utilities, SW = GA(b[:q], pctr[:q], pos_norm[:q], q, constant, len_content[:q], max_len)
#     print(SW)
