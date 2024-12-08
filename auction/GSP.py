import numpy as np


def GSP(b, pctr, pos_norm, q, constant, len_content, max_len):
    """
    Implements the Generalized Second Price (GSP) auction for allocating ad slots.

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
            Equal prominence (1/q) for each ad slot.
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

    for k in range(q):
        # Compute normalized eCPM adjusted by position
        ecpm_k[k] = b[sigma[k]] * pctr[sigma[k]] * pos_norm[k]
        # Assign equal prominence (1/q) for all allocated slots
        Prom[k] = 1 / q

    # Calculate payments for each ad
    payments = np.zeros(q)
    for k in range(q - 1):
        # Each ad pays the eCPM of the next highest bid
        payments[k] = ecpm[sigma[k + 1]]
    payments[q - 1] = 0  # Last slot pays nothing as there is no lower-ranked ad

    # Calculate utilities and social welfare
    utilities = np.zeros(q)
    SW = 0

    for k in range(q):
        # Utility = bid value * prominence - payment
        utilities[k] = b[sigma[k]] * Prom[k] - payments[k]
        # Social welfare includes normalized position effect
        SW += b[sigma[k]] * pctr[sigma[k]] * pos_norm[k] * (np.log(Prom[k]) + constant)

    return sigma, Prom, payments, utilities, SW
