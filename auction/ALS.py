import numpy as np

DEFAULT_BETA = 0.5


def _allocation(scores, positions, q, alpha):
    order = np.argsort(-scores, kind="stable")
    weights = scores[order[:q]] * positions[:q]
    scale = float(np.max(weights))
    if scale == 0:
        return order, np.full(q, 1 / q), 0.0
    powered = (weights / scale) ** alpha
    total = float(np.sum(powered))
    return order, powered / total, scale * total ** (1 / alpha)


def ALS(b, pctr, pos_norm, q, constant, len_content, max_len, *, beta=DEFAULT_BETA):
    b, pctr, positions = (np.asarray(v, dtype=float) for v in (b, pctr, pos_norm))
    if b.ndim != 1 or pctr.shape != b.shape or not len(b):
        raise ValueError("b and pctr must be nonempty one-dimensional arrays of equal length")
    if isinstance(q, (bool, np.bool_)) or not isinstance(q, (int, np.integer)) or not 1 <= q <= len(b):
        raise ValueError("q must be an integer between 1 and the number of advertisers")
    if not np.all(np.isfinite(b)) or np.any(b < 0):
        raise ValueError("DWLS requires finite nonnegative bids")
    if not np.all(np.isfinite(pctr)) or np.any(pctr <= 0) or np.any(pctr > 1):
        raise ValueError("base CTRs must be finite and in (0,1]")
    if (positions.ndim != 1 or len(positions) < q
            or not np.all(np.isfinite(positions[:q]))
            or np.any(positions[:q] <= 0) or np.any(positions[:q] > 1)
            or np.any(np.diff(positions[:q]) > 0)):
        raise ValueError("position discounts must be positive, at most 1 and non-increasing")
    if isinstance(beta, (bool, np.bool_)) or not np.isfinite(beta) or not 0 < beta < 1:
        raise ValueError("als_beta must be finite and strictly between 0 and 1")
    if not np.isfinite(constant):
        raise ValueError("constant must be finite")
    alpha = 1 / (1 - beta)
    scores = b * pctr
    sigma, prominence, native_welfare = _allocation(scores, positions, q, alpha)
    if np.any((prominence <= 0) & (scores[sigma[:q]] > 0)):
        raise ValueError("DWLS shares underflowed; reduce bid dynamic range or als_beta")
    final_ctr = pctr[sigma[:q]] * positions[:q] * prominence ** beta
    values = b[sigma[:q]] * final_ctr
    payments = np.zeros(q)
    for k, i in enumerate(sigma[:q]):
        others = np.delete(scores, i)
        if len(others) and np.any(others > 0):
            _, _, without_i = _allocation(others, positions, min(q, len(others)), alpha)
        else:
            without_i = 0.0
        payments[k] = np.clip(without_i - (native_welfare - values[k]), 0, values[k])
    utilities = values - payments
    # Use 0*log(0) = 0 for zero shares.
    log_share = np.zeros(q)
    np.log(prominence, out=log_share, where=prominence > 0)
    common_sw = float(np.sum(scores[sigma[:q]] * positions[:q]
                             * (log_share + constant)))
    return sigma, prominence, payments, utilities, common_sw
