import numpy as np


def GSP_SQA(b, pctr, pos_norm, q, constant, len_content, max_len, *, user_utilities=None):
    b, pctr, positions, lengths = (np.asarray(x, dtype=float) for x in (b,pctr,pos_norm,len_content))
    if b.ndim != 1 or len(b) < 2 or pctr.shape != b.shape or lengths.shape != b.shape:
        raise ValueError("SQA requires at least two advertisers and equal-length bid/CTR/length vectors")
    if isinstance(q,(bool,np.bool_)) or not isinstance(q,(int,np.integer)) or not 1 <= q <= len(b):
        raise ValueError("q must be an integer between 1 and the number of advertisers")
    if not np.all(np.isfinite(b)) or np.any(b < 0):
        raise ValueError("bids must be finite and nonnegative")
    if not np.all(np.isfinite(pctr)) or np.any(pctr <= 0) or np.any(pctr > 1):
        raise ValueError("base CTRs must lie in (0,1]")
    if positions.ndim != 1 or not len(positions) or not np.isfinite(positions[0]) or positions[0] <= 0:
        raise ValueError("the displayed position factor must be finite and positive")
    if not np.isfinite(max_len) or max_len < 1 or not np.all(np.isfinite(lengths)) or np.any(lengths < 1):
        raise ValueError("SQA needs nonempty ad creatives and a positive text budget")
    if not np.isfinite(constant):
        raise ValueError("constant must be finite")
    users = np.zeros(len(b)) if user_utilities is None else np.asarray(user_utilities,dtype=float)
    if users.shape != b.shape or not np.all(np.isfinite(users)):
        raise ValueError("provide one finite, bid-independent user utility per sponsored answer")
    sigma = np.argsort(-(b + users),kind="stable")
    winner, second = sigma[:2]
    prominence, payments, utilities = (np.zeros(q) for _ in range(3))
    prominence[0] = min(lengths[winner] / max_len,1.)
    payments[0] = b[second] + users[second] - users[winner]
    utilities[0] = b[winner] - payments[0]
    welfare = float(b[winner] * pctr[winner] * positions[0]
                    * (np.log(prominence[0]) + constant))
    return sigma,prominence,payments,utilities,welfare
