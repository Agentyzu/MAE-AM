import numpy as np


def perturbed_ranking(bids, relevance, q, noise):
    bids, relevance, noise = (np.asarray(x, dtype=float) for x in (bids, relevance, noise))
    if bids.ndim != 1 or not len(bids) or relevance.shape != bids.shape or noise.shape != bids.shape:
        raise ValueError("bids, relevance and Gumbel noise must be equal-length vectors")
    if isinstance(q, (bool, np.bool_)) or not isinstance(q, (int, np.integer)) or not 1 <= q <= len(bids):
        raise ValueError("q must be an integer between 1 and the number of advertisers")
    if not np.all(np.isfinite(bids)) or np.any(bids < 0):
        raise ValueError("bids must be finite and nonnegative")
    if not np.all(np.isfinite(relevance)) or np.any(relevance <= 0):
        raise ValueError("retrieval relevance must be finite and positive")
    if not np.all(np.isfinite(noise)):
        raise ValueError("Gumbel noise must be finite")
    # Log space avoids overflow and underflow.
    log_bids = np.full(len(bids), -np.inf)
    np.log(bids, out=log_bids, where=bids > 0)
    log_weights = np.log(relevance) + noise
    log_scores = log_bids + log_weights
    sigma = np.argsort(-log_scores, kind="stable")
    prices = np.zeros(q)
    if q < len(bids) and np.isfinite(log_scores[sigma[q]]):
        prices = np.exp(log_scores[sigma[q]] - log_weights[sigma[:q]])
        prices = np.minimum(prices, bids[sigma[:q]])  # Clamp threshold roundoff.
    return sigma, prices, log_scores


def AAL(b, pctr, pos_norm, q, constant, len_content, max_len, *, relevance=None, noise=None):
    b, pctr, positions = (np.asarray(x, dtype=float) for x in (b, pctr, pos_norm))
    if pctr.shape != b.shape or not np.all(np.isfinite(pctr)) or np.any(pctr <= 0) or np.any(pctr > 1):
        raise ValueError("base CTRs must match bids and lie in (0,1]")
    if relevance is None:
        relevance = pctr
    if noise is None:
        noise = np.random.gumbel(size=len(b))
    sigma, prices, _ = perturbed_ranking(b, relevance, q, noise)
    if (positions.ndim != 1 or len(positions) < q or not np.all(np.isfinite(positions[:q]))
            or np.any(positions[:q] <= 0) or np.any(np.diff(positions[:q]) > 0)):
        raise ValueError("position factors must be positive and non-increasing")
    if not np.isfinite(constant):
        raise ValueError("constant must be finite")
    prominence = np.full(q, 1 / q)
    winners = sigma[:q]
    payments = pctr[winners] * prices
    utilities = pctr[winners] * (b[winners] - prices)
    # Common platform objective at the AAL allocation.
    welfare = float(np.sum(b[winners] * pctr[winners] * positions[:q]
                           * (np.log(prominence) + constant)))
    return sigma, prominence, payments, utilities, welfare
