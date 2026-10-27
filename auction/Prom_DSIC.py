import numpy as np


def Prom_DSIC(b, pctr, pos_norm, q, constant, len_content, max_len, beta, s):
    """
    Tests bid misreporting in the prominence-based auction; values in b remain truthful.
    Zero reported slot weight yields no allocation. An absent competitor yields zero payment.

    Parameters:
    -----------
    b : np.ndarray
        Bid values for each advertisement.
    pctr : np.ndarray
        Predicted click-through rate (pCTR) for each advertisement.
    pos_norm : np.ndarray
        Position normalization factors for ad slots.
    q : int
        Number of available ad slots.
    constant : float
        A constant used in the social welfare calculation.
    len_content : np.ndarray
        Content length of each advertisement (unused in this implementation).
    max_len : int
        Maximum length available for all advertisements combined (unused in this implementation).
    beta : float
        Bid adjustment factor for testing DSIC (dominant strategy incentive compatibility).
    s : int
        Fixed advertiser index in the input arrays, not a rank position; its bid is multiplied by beta.

    Returns:
    --------
    tuple:
        - sigma : np.ndarray
            Indices of ads sorted by their expected cost per mille (eCPM) in descending order.
        - Prom : np.ndarray
            Prominence values for each ad slot based on normalized eCPM.
        - payments : np.ndarray
            Payment amount for each ad.
        - utilities : np.ndarray
            Utility for each allocated ad.
        - SW : float
            Social welfare value for the given allocation.
    """
    b_hat = b.copy()
    b_hat[s] = b[s] * beta
    ecpm = b_hat * pctr
    sigma = np.argsort(-ecpm)

    # Initialize prominence and normalized eCPM for q slots
    Prom = np.zeros(q)
    ecpm_k = np.zeros(q)

    # Calculate normalized eCPM values adjusted by position normalization
    for k in range(q):
        ecpm_k[k] = b_hat[sigma[k]] * pctr[sigma[k]] * pos_norm[k]

    payments = np.zeros(q)
    utilities = np.zeros(q)
    total_ecpm = np.sum(ecpm_k)
    if total_ecpm == 0:
        return sigma, Prom, payments, utilities, 0.0
    Prom = ecpm_k / total_ecpm

    if q == 1:
        if len(sigma) > 1:
            payments[0] = ecpm[sigma[1]]
    else:
        for k in np.flatnonzero(Prom > 0):
            w_minus_i = np.sum(ecpm_k[:k]) + np.sum(ecpm_k[k + 1:])
            if w_minus_i > 0:
                t = ecpm_k[k] / w_minus_i
                payments[k] = (w_minus_i / (pctr[sigma[k]] * pos_norm[k])) * (np.log1p(t) - t / (1 + t))

    for k in range(q):
        utilities[k] = b[sigma[k]] * Prom[k] - payments[k]

    true_ecpm_k = b[sigma[:q]] * np.asarray(pctr)[sigma[:q]] * np.asarray(pos_norm)[:q]
    allocated = Prom > 0
    SW = np.sum(true_ecpm_k[allocated] * (np.log(Prom[allocated]) + constant))

    return sigma, Prom, payments, utilities, SW
