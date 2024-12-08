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
            Payment amount for each ad.
        - utilities : np.ndarray
            Utility for each allocated ad.
        - SW : float
            Social welfare value for the given allocation.
    """
    # Calculate eCPM for each ad and sort indices in descending order
    ecpm = b * pctr
    sigma = np.argsort(-ecpm)

    # Initialize prominence and normalized eCPM for q slots
    Prom = np.zeros(q)
    ecpm_k = np.zeros(q)

    # Calculate prominence (Prom) based on ad content length
    for k in range(q):
        ecpm_k[k] = b[sigma[k]] * pctr[sigma[k]] * pos_norm[k]
        if k == 0:
            # Prominence for the first slot is proportional to content length, capped at 1
            Prom[k] = len_content[sigma[k]] / max_len
            Prom[k] = min(Prom[k], 1)  # Ensure Prom[k] does not exceed 1
        else:
            # Minimum prominence for subsequent slots
            Prom[k] = 0.01

    # Calculate payments for each ad
    payments = np.zeros(q)
    for k in range(q):
        if k == 0:
            # First slot pays the eCPM of the next highest ad
            payments[k] = ecpm[sigma[k + 1]]
        else:
            # Subsequent slots have zero payment
            payments[k] = 0

    # Calculate utilities and social welfare
    utilities = np.zeros(q)
    SW = 0

    for k in range(q):
        # Utility = bid value * prominence - payment
        utilities[k] = b[sigma[k]] * Prom[k] - payments[k]
        if Prom[k] != 0.01:
            # Include in social welfare calculation only if prominence is above the minimum
            SW += b[sigma[k]] * pctr[sigma[k]] * pos_norm[k] * (np.log(Prom[k]) + constant)

    return sigma, Prom, payments, utilities, SW
