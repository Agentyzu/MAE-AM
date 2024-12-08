import numpy as np


def Prom(b, pctr, pos_norm, q, constant, len_content, max_len):
    """
    Implements a prominence-based allocation mechanism for ads.

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
    # Calculate eCPM for each ad and sort indices in descending order
    ecpm = b * pctr
    sigma = np.argsort(-ecpm)

    # Initialize prominence and normalized eCPM for q slots
    Prom = np.zeros(q)
    ecpm_k = np.zeros(q)

    # Calculate normalized eCPM values adjusted by position normalization
    for k in range(q):
        ecpm_k[k] = b[sigma[k]] * pctr[sigma[k]] * pos_norm[k]

    # Calculate prominence values based on normalized eCPM
    for k in range(q):
        Prom[k] = ecpm_k[k] / np.sum(ecpm_k)

    # Initialize payments array
    payments = np.zeros(q)

    # Calculate payments based on the prominence allocation
    if q == 1:
        # Single ad case: pay the eCPM of the next highest ad (edge case)
        payments[0] = ecpm[sigma[1]]
    else:
        for k in range(q):
            # Calculate the payment using the prominence-based formula
            w_minus_i = np.sum(ecpm_k) - ecpm_k[k]
            t = ecpm_k[k] / w_minus_i
            payments[k] = (w_minus_i / (pctr[sigma[k]] * pos_norm[k])) * (np.log(1 + t) - t / (1 + t))

    # Calculate utilities for each ad
    utilities = np.zeros(q)
    for k in range(q):
        utilities[k] = b[sigma[k]] * Prom[k] - payments[k]

    # Calculate social welfare using normalized eCPM and prominence values
    SW = np.sum(ecpm_k * (np.log(ecpm_k / np.sum(ecpm_k)) + constant))

    return sigma, Prom, payments, utilities, SW
