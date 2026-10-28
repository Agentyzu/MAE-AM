import numpy as np

from auction.Prom import Prom


def Prom_DSIC(b, pctr, pos_norm, q, constant, len_content, max_len, beta, s):
    values = np.asarray(b, dtype=float)
    ctr = np.asarray(pctr, dtype=float)
    if values.ndim != 1 or ctr.shape != values.shape:
        raise ValueError("b and pctr must be one-dimensional arrays of equal length")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("true values must be finite and nonnegative")
    if not isinstance(s, (int, np.integer)) or not 0 <= s < len(values):
        raise ValueError("s must be a valid zero-based rank in the original ordering")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")

    original_rank = np.argsort(-(values * ctr), kind="stable")
    reported_bids = values.copy()
    target = original_rank[s]
    reported_bids[target] = values[target] * beta

    sigma, prominence, payments, _, SW = Prom(
        reported_bids, ctr, pos_norm, q, constant, len_content, max_len
    )
    utilities = values[sigma[:q]] * prominence - payments
    return sigma, prominence, payments, utilities, SW
