from copy import deepcopy
import numpy as np


def shortlist(bids, relevance, k):
    bids, relevance = np.asarray(bids, float), np.asarray(relevance, float)
    if (bids.ndim != 1 or not len(bids) or bids.shape != relevance.shape
            or not np.all(np.isfinite(bids)) or np.any(bids < 0)
            or not np.all(np.isfinite(relevance)) or np.any(relevance <= 0)):
        raise ValueError("LERA requires finite nonnegative bids and positive Stage 1 scores")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not 1 <= k <= len(bids):
        raise ValueError("LERA shortlist size must lie in [1,n]")
    rank = np.argsort(-(bids * relevance), kind="stable")
    # Keep prompt labels in advertiser order.
    return np.sort(rank[:k]), rank


def single_round(bids, relevance, choice_logprobs, k):
    candidates, rank = shortlist(bids, relevance, k)
    b, s, z = np.asarray(bids, float), np.asarray(relevance, float), np.asarray(choice_logprobs, float)
    if z.shape != (k + 1,) or not np.all(np.isfinite(z)):
        raise ValueError("LERA needs actual finite log-probabilities for all K choices plus no insertion")
    log_b = np.full(k, -np.inf)
    np.log(b[candidates], out=log_b, where=b[candidates] > 0)
    log_scores = z + np.r_[0., log_b]  # No insertion has bid 1.
    choice = int(np.argmax(log_scores))  # Ties prefer no insertion, then advertiser ID.
    probabilities = np.exp(z - np.max(z)); probabilities /= probabilities.sum()
    result = dict(candidate_indices=candidates.tolist(), winner=None, per_click_price=0.,
                  entry_CPC=0., ranking_CPC=0., choice_probabilities=probabilities.tolist())
    if choice:
        winner = int(candidates[choice - 1])
        entry = float(b[rank[k]] * s[rank[k]] / s[winner]) if k < len(b) else 0.
        second = np.max(np.delete(log_scores, choice))
        ranking = float(np.exp(second - z[choice]))
        price = max(entry, ranking)
        if price > b[winner] + 1e-10 * max(1., b[winner]):
            raise RuntimeError("LERA critical CPC exceeds winning bid beyond roundoff")
        result.update(winner=winner, entry_CPC=entry, ranking_CPC=ranking,
                      per_click_price=min(price, float(b[winner])))
    return result


def LERA(b, pctr, pos_norm, q, constant, len_content, max_len, *, rounds=None, details=None):
    b, ctr, pos = (np.asarray(v, float) for v in (b, pctr, pos_norm))
    if (b.ndim != 1 or not len(b) or ctr.shape != b.shape
            or not np.all(np.isfinite(b)) or np.any(b < 0)
            or not np.all(np.isfinite(ctr)) or np.any(ctr < 0) or np.any(ctr > 1)):
        raise ValueError("LERA bids and CTRs must be finite equal-length vectors")
    if (isinstance(q, (bool, np.bool_)) or not isinstance(q, (int, np.integer)) or not 1 <= q <= len(b)
            or pos.ndim != 1 or len(pos) < q or not np.all(np.isfinite(pos[:q]))
            or np.any(pos[:q] <= 0) or np.any(np.diff(pos[:q]) > 0) or not np.isfinite(constant)):
        raise ValueError("Invalid LERA capacity, position factors or welfare constant")
    if rounds is None:
        raise ValueError("LERA requires recorded two-stage rounds; use ModelZoo with an SGLang scoring service")
    selected, cpcs, outcomes = [], [], []
    for step in rounds:
        if len(selected) == q:
            raise ValueError("LERA rounds exceed configured insertion capacity")
        eligible = np.asarray(step["eligible_indices"], dtype=int)
        expected = [i for i in range(len(b)) if i not in selected]
        if eligible.tolist() != expected:
            raise ValueError("LERA eligible advertiser order changed or repeats an inserted advertiser")
        outcome = single_round(b[eligible], step["relevance"], step["choice_logprobs"], step["k"])
        winner = None if outcome["winner"] is None else int(eligible[outcome["winner"]])
        if winner != step["winner"]:
            raise ValueError("Saved LERA prefix winner disagrees with its auction scores")
        outcomes.append(dict(outcome, winner=winner, candidate_indices=eligible[outcome["candidate_indices"]].tolist()))
        if winner is not None:
            selected.append(winner); cpcs.append(outcome["per_click_price"])
    count = len(selected)
    sigma = np.array(selected + [i for i in range(len(b)) if i not in selected], dtype=int)
    shares, paid, utility = (np.zeros(q) for _ in range(3))
    if count:
        shares[:count] = 1 / count
        paid[:count] = ctr[selected] * cpcs
        utility[:count] = ctr[selected] * (b[selected] - cpcs)
    sw = float(np.sum(b[selected] * ctr[selected] * pos[:count]
                      * (np.log(shares[:count]) + constant))) if count else 0.
    if details is not None:
        all_paid, all_utility = np.zeros(len(b)), np.zeros(len(b))
        all_paid[selected], all_utility[selected] = paid[:count], utility[:count]
        details.clear()
        details.update(mechanism="LERA_two_stage_critical_CPC", configured_capacity=int(q),
                       displayed_ad_count=count, assigned_advertisers=selected, rounds=deepcopy(outcomes),
                       per_click_prices=cpcs, per_advertiser_payments=all_paid.tolist(),
                       per_advertiser_utilities=all_utility.tolist(),
                       native_welfare=float(np.dot(b[selected], ctr[selected])),
                       payment_basis="shared_CTR_times_max_of_stage1_entry_and_stage2_winning_CPC",
                       utility_basis="shared_CTR_times_(value_minus_CPC)",
                       value_assumption="supplied_truthful_per_click_values_and_shared_expected_CTR_proxy",
                       rendering_order="sequential_native_insertion_order",
                       ad_prom_semantics="equal_word_budget_over_selected_ads",
                       reported_SW="common_log_platform_objective_at_rendered_winner_set")
    return sigma, shares, paid, utility, sw
