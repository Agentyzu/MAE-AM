from copy import deepcopy
from functools import partial
from math import fsum

import numpy as np

from auction.GSP_SQA import GSP_SQA
from utils.text_length import count_text
from utils.experiment_inputs import apply, stamp


_FIELDS = ("query", "basic_reply", "ad_name", "content", "b", "pctr", "pos_norm",
           "q", "max_len", "constant", "sigma", "prom", "payments", "utilities",
           "SW", "ad_len", "basic_reply_provenance", "dataset", "language", "length_unit", "length_counting_rule")


def _snapshot(model):
    return {name: deepcopy(getattr(model, name)) for name in _FIELDS}


def prepare(processes):
    if not processes:
        raise ValueError("Configure at least one supported SQA candidate model")
    first = next(iter(processes.values()))
    model = first.llm
    model.messages = []
    paired = apply(model, include_bids=False)
    if not paired:
        call_start = len(model.llm_calls)
        model.basic_reply = model.reply(model.organic_prompt())
        model.basic_reply_provenance = deepcopy(model.llm_calls[call_start:])
    n = len(model.content)
    if n < 2:
        raise ValueError("The SQA source requires at least two advertisers")
    lengths = np.array([count_text(text, model.length_unit) for text in model.content])
    if np.any(lengths < 1) or model.max_len < 1:
        raise ValueError("SQA requires nonempty creatives and a positive ad budget")
    # Freeze sponsored answers before bidding.
    model.b = np.zeros(n)
    model.q = 1
    model.SW = 0.0
    model.payments = model.utilities = np.zeros(1)
    context = _snapshot_context(model)
    bank, fixed = {}, []
    for i in range(n):
        request = deepcopy(context)
        request.update(sigma=np.array([i] + [j for j in range(n) if j != i]),
                       prom=np.array([min(lengths[i] / model.max_len, 1.)]))
        request["ad_len"] = [int(model.max_len * request["prom"][0])]
        bank[i] = {}
        for name, process in processes.items():
            process.run(request, generation_style="sqa")
            bank[i][name] = {"result": deepcopy(process.result),
                             "messages": deepcopy(process.llm.messages)}
        chosen = max(processes, key=lambda name: bank[i][name]["result"]["mean_score"])
        result = bank[i][chosen]["result"]
        sample = result["samples"][result["best_sample_index"]]
        spans = sample["ad_spans"]
        if (len(spans) != 1 or spans[0]["advertiser_index"] != i
                or spans[0]["status"] != "ok" or not spans[0]["text"].strip()):
            raise ValueError(f"SQA advertiser {i} has no valid ad span in its selected fixed answer")
        fixed.append({"advertiser_index": i, "selected_model": chosen,
                      "selected_sample_index": sample["sample_index"],
                      "user_utility": sample["user_satis"], "gen_reply": sample["gen_reply"],
                      "model_mean_semantic_scores": {
                          name: bank[i][name]["result"]["mean_score"] for name in processes}})
    users = np.array([item["user_utility"] for item in fixed])
    if paired:
        model.b = np.array(model.shared_instance["bids"], dtype=float)
    else:
        model.calculate_similar(model.basic_reply)
    model.auc_alg = partial(GSP_SQA, user_utilities=users)
    model.auction()
    shared = _snapshot(model)
    shared["sqa_setup"] = {
        "configured_capacity": first.requested_q, "displayed_ad_count": 1,
        "candidate_models": list(processes), "n_samples_per_model_per_ad": first.n_samples,
        "fusion_timing": "all_ad_answers_fixed_before_bidding",
        "user_utility_proxy": "BGE_cosine_fixed_sponsored_answer_vs_organic_answer",
        "preauction_SW_for_semantic_selection": 0.0,
        "fixed_answers": fixed, "scorer": model.scorer.metadata()
    }
    return shared, bank


def _snapshot_context(model):
    # Use bid-independent preauction inputs.
    return {name: deepcopy(getattr(model, name)) for name in _FIELDS
            if name not in ("sigma", "prom", "ad_len")}


def use_fixed_answer(process, shared_request=None, candidate=None):
    if shared_request is None:
        shared_request, bank = prepare({"model": process})
        candidate = bank[int(shared_request["sigma"][0])]["model"]
    if candidate is None:
        raise ValueError("SQA needs the candidate bank associated with its preauction answers")
    model = process.llm
    for name, value in shared_request.items():
        setattr(model, name, deepcopy(value))
    process.basic_reply = model.basic_reply
    result = deepcopy(candidate["result"])
    delta = model.theta_2 * model.SW
    for sample in result["samples"]:
        sample["score"] += delta
    result["score"] = result["mean_score"] = fsum(s["score"] for s in result["samples"]) / len(result["samples"])
    best = result["samples"][result["best_sample_index"]]
    result["best_sample_score"] = best["score"]
    result.update(ad_bids=np.asarray(model.b).tolist(), ad_pctr=np.asarray(model.pctr).tolist(),
                  ad_rank=model.sigma[:1].tolist(), ad_prom=model.prom[:1].tolist(),
                  ad_rev=model.payments[:1].tolist(), ad_utility=model.utilities[:1].tolist(),
                  ad_sw=model.SW, ad_len=list(model.ad_len[:1]), sqa_setup=deepcopy(model.sqa_setup))
    winner = int(model.sigma[0])
    users = np.array([item["user_utility"] for item in model.sqa_setup["fixed_answers"]])
    paid, utility = np.zeros(len(model.b)), np.zeros(len(model.b))
    paid[winner], utility[winner] = model.payments[0], model.utilities[0]
    result["auction"] = {
        "mechanism": "SQA_single_fixed_sponsored_answer", "displayed_ad_count": 1,
        "platform_values": (np.asarray(model.b) + users).tolist(),
        "user_utilities": users.tolist(), "payment_basis": "per_sponsored_answer_display",
        "utility_basis": "value_of_display_minus_payment_for_winner_zero_for_losers",
        "value_assumption": "supplied_bids_equal_display_values_for_utility_reporting",
        "native_welfare": float(model.b[winner] + users[winner]),
        "per_advertiser_payments": paid.tolist(), "per_advertiser_utilities": utility.tolist(),
        "signed_payment_policy": "unclipped_source_formula_may_include_subsidy",
        "reported_SW": "common_log_platform_objective_at_single_displayed_ad"
    }
    stamp(result, model)
    process.result = result
    process._report_target_utilities()
    process.gen_reply = model.gen_reply = model.compressed_ad = best["gen_reply"]
    model.messages = deepcopy(candidate["messages"])
    for key in ("raw_gen_reply", "ad_spans", "span_diagnostics", "scoring",
                "user_satis", "ad_satis", "score"):
        setattr(model, key, deepcopy(best[key]))
    model._span_reply = best["gen_reply"]
    return process.gen_reply
