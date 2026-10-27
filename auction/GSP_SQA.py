import numpy as np


def GSP_SQA(b, pctr, pos_norm, q, constant, len_content, max_len):
    """
    Implements a SQA auction with a slot quality adjustment (GSP_SQA).

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
        Content length of each advertisement.
    max_len : int
        Maximum length available for all advertisements combined.

    Returns:
    --------
    tuple:
        - sigma : np.ndarray
            Indices of ads sorted by their expected cost per mille (eCPM) in descending order.
        - Prom : np.ndarray
            Prominence values for each ad slot based on content length and slot adjustments.
        - payments : np.ndarray
            Slot-aligned payments, with zero for unallocated slots.
        - utilities : np.ndarray
            Slot-aligned utilities, with zero for unallocated slots.
        - SW : float
            Social welfare value for the given allocation.
    """
    # Calculate eCPM for each ad and sort indices in descending order
    ecpm = b * pctr
    sigma = np.argsort(-ecpm)

    # Initialize prominence and normalized eCPM for q slots
    Prom = np.zeros(q)
    ecpm_k = np.zeros(q)

    if q > 0 and max_len > 0:
        Prom[0] = min(len_content[sigma[0]] / max_len, 1)

    allocated = np.flatnonzero(Prom > 0)
    payments = np.zeros(q)
    utilities = np.zeros(q)
    for k in allocated:
        ecpm_k[k] = ecpm[sigma[k]] * pos_norm[k]
        if k + 1 < len(sigma):
            payments[k] = ecpm[sigma[k + 1]]
        utilities[k] = b[sigma[k]] * Prom[k] - payments[k]
    SW = np.sum(ecpm_k[allocated] * (np.log(Prom[allocated]) + constant))

    return sigma, Prom, payments, utilities, SW
