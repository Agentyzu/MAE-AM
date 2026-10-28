import numpy as np


def _myerson_payment(i, bid, ctr, ecpm, pos_norm, q, share):
    thresholds = np.delete(ecpm, i) / ctr
    thresholds = thresholds[(thresholds > 0) & (thresholds < bid)]
    cuts = np.unique(np.concatenate(([0.0], thresholds, [bid])))
    integral = 0.0

    for lower, upper in zip(cuts[:-1], cuts[1:]):
        # Ranks are fixed between bid thresholds.
        scores = ecpm.copy()
        scores[i] = (lower + (upper - lower) / 2) * ctr
        winners = np.argsort(-scores, kind="stable")[:q]
        slot = np.flatnonzero(winners == i)
        if slot.size == 0:
            continue

        k = int(slot[0])
        alpha = ctr * pos_norm[k]
        other = winners != i
        w_minus_i = np.sum(scores[winners[other]] * pos_norm[:q][other])
        width = upper - lower
        if w_minus_i == 0:
            contribution = width
        else:
            # log1p preserves precision on narrow intervals.
            contribution = width - (w_minus_i / alpha) * np.log1p(
                alpha * width / (w_minus_i + alpha * lower)
            )
        integral += max(0.0, contribution)

    # Clamp negative roundoff.
    return max(0.0, bid * share - integral)


def Prom(b, pctr, pos_norm, q, constant, len_content, max_len):
    b = np.asarray(b, dtype=float)
    pctr = np.asarray(pctr, dtype=float)
    pos_norm = np.asarray(pos_norm, dtype=float)
    if b.ndim != 1 or pctr.shape != b.shape:
        raise ValueError("b and pctr must be one-dimensional arrays of equal length")
    if not isinstance(q, (int, np.integer)) or not 1 <= q <= len(b):
        raise ValueError("q must be an integer between 1 and the number of advertisers")
    if not np.all(np.isfinite(b)) or np.any(b < 0):
        raise ValueError("bids must be finite and nonnegative")
    if not np.all(np.isfinite(pctr)) or np.any(pctr <= 0):
        raise ValueError("CTRs must be finite and positive")
    if (pos_norm.ndim != 1 or len(pos_norm) < q
            or not np.all(np.isfinite(pos_norm[:q]))
            or np.any(pos_norm[:q] <= 0) or np.any(np.diff(pos_norm[:q]) > 0)):
        raise ValueError("position biases must be positive and non-increasing for all q slots")
    if np.count_nonzero(b > 0) < q:
        raise ValueError("GPA requires at least q positive bids")

    # Bid ties use advertiser input order.
    ecpm = b * pctr
    sigma = np.argsort(-ecpm, kind="stable")

    Prom = np.zeros(q)
    ecpm_k = np.zeros(q)

    for k in range(q):
        ecpm_k[k] = b[sigma[k]] * pctr[sigma[k]] * pos_norm[k]

    for k in range(q):
        Prom[k] = ecpm_k[k] / np.sum(ecpm_k)

    payments = np.zeros(q)

    # Integrate across entry and rank changes.
    if q == 1:
        payments[0] = ecpm[sigma[1]] / pctr[sigma[0]] if len(b) > 1 else 0.0
    else:
        for k in range(q):
            i = sigma[k]
            payments[k] = _myerson_payment(
                i, b[i], pctr[i], ecpm, pos_norm, q, Prom[k]
            )

    utilities = np.zeros(q)
    for k in range(q):
        utilities[k] = b[sigma[k]] * Prom[k] - payments[k]

    SW = np.sum(ecpm_k * (np.log(ecpm_k / np.sum(ecpm_k)) + constant))

    return sigma, Prom, payments, utilities, SW
