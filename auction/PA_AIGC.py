import numpy as np


def mnl_allocation(bids, position_ctrs):
    from scipy.optimize import linprog

    bids, p = np.asarray(bids, dtype=float), np.asarray(position_ctrs, dtype=float)
    if bids.ndim != 1 or p.ndim != 2 or p.shape[0] != len(bids) or not p.shape[1]:
        raise ValueError("Position CTRs must be an n-by-q matrix in advertiser order")
    if not np.all(np.isfinite(bids)) or np.any(bids < 0):
        raise ValueError("Bids must be finite and nonnegative")
    if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p >= 1):
        raise ValueError("Standalone position CTRs must lie in [0,1)")
    n, q = p.shape
    empty = (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float), 0.)
    if not n or not np.any(bids[:, None] * p > 0):
        return empty
    odds = p / (1 - p)
    size = n * q
    constraints = np.zeros((n + q + 1, size + 1))
    for i in range(n):
        constraints[i, i*q:(i+1)*q] = 1
    for j in range(q):
        constraints[n+j, j:size:q] = 1
    constraints[:-1, -1] = -1
    constraints[-1, :size], constraints[-1, -1] = 1, -q
    scale = float(np.max(bids))
    objective = np.r_[-((bids / scale)[:, None] * odds).ravel(), 0.]
    result = linprog(objective, A_ub=constraints, b_ub=np.zeros(n+q+1),
                     A_eq=np.r_[odds.ravel(), 1.][None, :], b_eq=[1.],
                     bounds=(0, None), method="highs-ds",
                     options={"primal_feasibility_tolerance": 1e-9,
                              "dual_feasibility_tolerance": 1e-9})
    if not result.success or result.x[-1] <= 0:
        raise RuntimeError(f"PA-AIGC MNL LP failed: {result.message}")
    matching = result.x[:size].reshape(n, q) / result.x[-1]
    integral = np.rint(matching)
    if (not np.allclose(matching, integral, rtol=0, atol=1e-6)
            or np.any(integral < 0) or np.any(integral > 1)
            or np.any(integral.sum(axis=0) > 1) or np.any(integral.sum(axis=1) > 1)):
        raise RuntimeError("PA-AIGC LP did not yield a feasible integral matching")
    advertisers, positions = np.nonzero(integral)
    keep = odds[advertisers, positions] > 0
    advertisers, positions = advertisers[keep], positions[keep]
    order = np.argsort(positions, kind="stable")
    advertisers, positions = advertisers[order], positions[order]
    weight = odds[advertisers, positions]
    clicks = weight / (1 + weight.sum())
    welfare = float(np.dot(bids[advertisers], clicks))
    if not np.isclose(welfare / scale, -result.fun, atol=1e-8, rtol=1e-8):
        raise RuntimeError("Recovered PA-AIGC matching disagrees with LP objective")
    return advertisers, positions, clicks, welfare


def PA_AIGC(b, pctr, pos_norm, q, constant, len_content, max_len, *,
            position_ctrs=None, details=None):
    b, pctr, factors = (np.asarray(x, dtype=float) for x in (b, pctr, pos_norm))
    if b.ndim != 1 or not len(b) or pctr.shape != b.shape:
        raise ValueError("Bids and base CTRs must be nonempty equal-length vectors")
    if isinstance(q, (bool, np.bool_)) or not isinstance(q, (int, np.integer)) or not 1 <= q <= len(b):
        raise ValueError("q must be an integer in [1,n]")
    if not np.all(np.isfinite(pctr)) or np.any(pctr < 0) or np.any(pctr >= 1):
        raise ValueError("Base CTRs must lie in [0,1)")
    if (factors.ndim != 1 or len(factors) < q or not np.all(np.isfinite(factors[:q]))
            or np.any(factors[:q] <= 0) or np.any(factors[:q] > 1)
            or np.any(np.diff(factors[:q]) > 0) or not np.isfinite(constant)):
        raise ValueError("Position factors must lie in (0,1] and be non-increasing; constant must be finite")
    p = pctr[:, None] * factors[None, :q] if position_ctrs is None else np.asarray(position_ctrs, dtype=float)
    if p.shape != (len(b), q):
        raise ValueError("Explicit position CTR matrix must have shape (n,q)")
    advertisers, positions, clicks, welfare = mnl_allocation(b, p)
    count = len(advertisers)
    shares, payments, utilities = (np.zeros(q) for _ in range(3))
    if count:
        shares[:count] = factors[positions] / factors[positions].sum()
    without = []
    tolerance = 1e-8 * max(1., float(np.max(b)))
    for k, i in enumerate(advertisers):
        remaining = np.arange(len(b)) != i
        _, _, _, counterfactual = mnl_allocation(b[remaining], p[remaining])
        payment = counterfactual - (welfare - b[i] * clicks[k])
        if payment < -tolerance or payment > b[i] * clicks[k] + tolerance:
            raise RuntimeError("PA-AIGC VCG payment violates nonnegativity or IR beyond solver tolerance")
        payments[k] = np.clip(payment, 0., b[i] * clicks[k])
        utilities[k] = b[i] * clicks[k] - payments[k]
        without.append(float(counterfactual))
    winners = set(advertisers.tolist())
    sigma = np.r_[advertisers, [i for i in range(len(b)) if i not in winners]].astype(int)
    common_sw = float(np.sum(b[advertisers] * pctr[advertisers] * factors[positions]
                              * (np.log(shares[:count]) + constant))) if count else 0.
    if details is not None:
        paid, utility, final_ctr = (np.zeros(len(b)) for _ in range(3))
        paid[advertisers], utility[advertisers], final_ctr[advertisers] = payments[:count], utilities[:count], clicks
        details.clear()
        details.update(
            mechanism="PA_AIGC_MNL_welfare_VCG", configured_capacity=int(q), displayed_ad_count=count,
            position_ctrs=p.tolist(), position_ctr_source=("shared_base_CTR_times_position_factor"
                            if position_ctrs is None else "explicit_position_CTR_matrix"),
            position_ids=positions.tolist(), assigned_advertisers=advertisers.tolist(),
            final_pctr=clicks.tolist(), per_advertiser_final_pctr=final_ctr.tolist(),
            per_advertiser_payments=paid.tolist(), per_advertiser_utilities=utility.tolist(),
            native_welfare=welfare, leave_one_out_welfare=without,
            payment_basis="VCG_expected_per_impression_MNL_click_externality",
            utility_basis="value_times_MNL_final_CTR_minus_expected_payment",
            value_assumption="supplied_bids_equal_per_click_values_for_utility_reporting",
            rendering_order="ascending_allocated_position_ID", solver="paper_LP_5_scipy_highs_ds",
            ad_prom_semantics="normalized_occupied_position_factors_for_text_rendering",
            reported_SW="common_log_platform_objective_using_shared_CTR_and_allocated_position_factors",
            empty_display_generation="organic_answer_copy_without_sponsored_LLM_calls")
    return sigma, shares, payments, utilities, common_sw
