import numpy as np


def GA(b, pctr, pos_norm, q, constant, len_content, max_len):
    """
    Implements a Greedy Auction (GA) algorithm for allocating ad slots and calculating payments and utilities.

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
            Normalized allocations (prominence) for each ad slot.
        - payments : np.ndarray
            Payment amount for each allocated ad.
        - utilities : np.ndarray
            Utility for each allocated ad.
        - SW : float
            Social welfare value for the given allocation.
    """
    # Calculate eCPM for each ad and sort indices in descending order of eCPM
    ecpm = b * pctr
    sigma = np.argsort(-ecpm)

    # Initialize prominence and eCPM after normalization for q slots
    Prom = np.zeros(q)
    ecpm_k = np.zeros(q)

    # Compute eCPM adjusted by position normalization for the top-q ads
    for k in range(q):
        ecpm_k[k] = b[sigma[k]] * pctr[sigma[k]] * pos_norm[k]

    # Allocate ad lengths based on available space
    remaining_len = max_len
    for k in range(q):
        idx = sigma[k]
        if remaining_len > 0:
            ad_len = len_content[idx]
            alloc_len = min(ad_len, remaining_len)  # Allocate as much as possible within the remaining space
            Prom[k] = alloc_len / max_len          # Normalize allocation
            remaining_len -= alloc_len
        else:
            Prom[k] = 0.01                         # Minimum prominence value

    # Calculate payments for allocated ads
    num_allocated_ads = np.sum(Prom > 0.01)       # Count ads with non-minimum prominence
    payments = np.zeros(num_allocated_ads)

    if num_allocated_ads == 1:
        # Special case: only one ad allocated
        payments[0] = ecpm[sigma[1]]
    else:
        # Calculate payment for each allocated ad
        for k in range(num_allocated_ads):
            w_minus_i = np.sum(ecpm_k) - ecpm_k[k]  # Total eCPM excluding the current ad
            t = ecpm_k[k] / w_minus_i               # Ratio for payment calculation
            payments[k] = (w_minus_i / (pctr[sigma[k]] * pos_norm[k])) * (
                np.log(1 + t) - t / (1 + t)
            )

    # Calculate utilities and social welfare
    utilities = np.zeros(num_allocated_ads)
    SW = 0

    for k in range(num_allocated_ads):
        # Utility = bid value * prominence - payment
        utilities[k] = b[sigma[k]] * Prom[k] - payments[k]

    for k in range(q):
        if Prom[k] != 0.01:  # Only consider allocated ads for social welfare
            SW += b[sigma[k]] * pctr[sigma[k]] * pos_norm[k] * (np.log(Prom[k]) + constant)

    return sigma, Prom, payments, utilities, SW
