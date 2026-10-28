import numpy as np


def GSP(b, pctr, pos_norm, q, constant, len_content, max_len):
    b = np.asarray(b, dtype=float)
    pctr = np.asarray(pctr, dtype=float)
    pos_norm = np.asarray(pos_norm, dtype=float)
    if b.ndim != 1 or pctr.shape != b.shape or not len(b):
        raise ValueError("b and pctr must be nonempty equal-length vectors")
    if isinstance(q, (bool, np.bool_)) or not isinstance(q, (int, np.integer)) or not 1 <= q <= len(b):
        raise ValueError("q must be an integer between 1 and the number of advertisers")
    if not np.all(np.isfinite(b)) or np.any(b < 0):
        raise ValueError("bids must be finite and nonnegative")
    if not np.all(np.isfinite(pctr)) or np.any(pctr <= 0) or np.any(pctr > 1):
        raise ValueError("base CTRs must be finite and in (0,1]")
    if (pos_norm.ndim != 1 or len(pos_norm) < q or not np.all(np.isfinite(pos_norm[:q]))
            or np.any(pos_norm[:q] <= 0) or np.any(np.diff(pos_norm[:q]) > 0)):
        raise ValueError("position factors must be positive and non-increasing")
    if not np.isfinite(constant):
        raise ValueError("constant must be finite")

    ecpm = b * pctr
    sigma = np.argsort(-ecpm)

    Prom = np.zeros(q)
    ecpm_k = np.zeros(q)

    for k in range(q):
        ecpm_k[k] = b[sigma[k]] * pctr[sigma[k]] * pos_norm[k]
        Prom[k] = 1 / q

    # The first loser sets the final slot price.
    payments = np.zeros(q)
    for k in range(q):
        if k + 1 < len(sigma):
            cpc = ecpm[sigma[k + 1]] / pctr[sigma[k]]
            slot_ctr = pctr[sigma[k]] * pos_norm[k]
            payments[k] = slot_ctr * cpc

    utilities = np.zeros(q)
    SW = 0

    for k in range(q):
        utilities[k] = b[sigma[k]] * pctr[sigma[k]] * pos_norm[k] - payments[k]
        SW += b[sigma[k]] * pctr[sigma[k]] * pos_norm[k] * (np.log(Prom[k]) + constant)

    return sigma, Prom, payments, utilities, SW
