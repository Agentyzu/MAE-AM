import numpy as np
from copy import deepcopy
from math import fsum, isfinite
from numbers import Integral
from auction.GSP import GSP
from auction.GA import GA
from auction.PA_AIGC import PA_AIGC
from auction.LERA import LERA
from utils.experiment_inputs import apply, stamp


class LLMProcess:
    def __init__(self, model_class, cfg, data, auc_alg):
        self.requested_q = cfg.q
        self.generation_config = {"q": cfg.q, "max_len": cfg.max_len,
                                  "is_structured": cfg.is_structured}
        self.als_beta = cfg.get("als_beta", 0.5)
        self.n_samples = cfg.get("n_samples", 5)
        if (isinstance(self.n_samples, bool)
                or not isinstance(self.n_samples, Integral) or self.n_samples < 1):
            raise ValueError("n_samples must be a positive integer")
        self.n_samples = int(self.n_samples)
        self.llm = model_class(cfg, data, auc_alg)
        if auc_alg is PA_AIGC:
            from functools import partial
            self.pa_aigc_state = {}
            self.pa_aigc_positions = self.llm.pos_norm.copy()
            self.llm.auc_alg = partial(PA_AIGC, position_ctrs=cfg.get("pa_aigc_position_ctrs"),
                                       details=self.pa_aigc_state)
        if auc_alg is LERA:
            from utils.lera_retrieval import settings, SGLangScorer
            self.lera_client = SGLangScorer(settings(cfg.get("lera", {}), cfg.q))
            self.lera_state = {}
            self.lera_positions = self.llm.pos_norm.copy()
        self.basic_reply = None
        self.gen_reply = None
        self.result = {}

    def prepare_request(self, *, auction_style="normal"):
        self.llm.messages = []
        if hasattr(self, "pa_aigc_state"):
            self.llm.q = self.requested_q
            self.llm.pos_norm = self.pa_aigc_positions.copy()
        if hasattr(self, "lera_state"):
            self.llm.q = self.requested_q
            self.llm.pos_norm = self.lera_positions.copy()
        call_start = len(self.llm.llm_calls)
        if apply(self.llm):
            self.basic_reply = self.llm.basic_reply
        else:
            self.basic_reply = self.llm.reply(self.llm.organic_prompt())
            self.llm.basic_reply = self.basic_reply
            self.llm.basic_reply_provenance = deepcopy(self.llm.llm_calls[call_start:])
            self.llm.calculate_similar(self.basic_reply)
        if auction_style == "aal":
            from functools import partial
            from auction.AAL import AAL
            from utils.aal_retrieval import retrieve
            self.llm.aal_retrieval = retrieve(self.llm)
            self.llm.aal_noise = np.random.gumbel(size=len(self.llm.b))
            self.llm.auc_alg = partial(AAL, relevance=self.llm.aal_retrieval["relevance"],
                                       noise=self.llm.aal_noise)
        elif auction_style != "normal":
            raise ValueError(f"Unknown auction preparation style: {auction_style}")
        if hasattr(self, "lera_state"):
            from functools import partial
            from utils.lera_retrieval import prepare
            self.llm.lera_retrieval = prepare(self.llm, self.lera_client)
            self.llm.auc_alg = partial(LERA, rounds=self.llm.lera_retrieval["rounds"], details=self.lera_state)
        self.llm.auction()
        if hasattr(self, "pa_aigc_state"):
            self.llm.pa_aigc_auction = deepcopy(self.pa_aigc_state)
            count = self.pa_aigc_state["displayed_ad_count"]
            self.llm.q = count
            self.llm.pos_norm = self.llm.pos_norm[self.pa_aigc_state["position_ids"]]
            for name in ("prom", "payments", "utilities", "ad_len"):
                setattr(self.llm, name, getattr(self.llm, name)[:count])
        if hasattr(self, "lera_state"):
            self.llm.lera_auction = deepcopy(self.lera_state)
            count = self.lera_state["displayed_ad_count"]
            self.llm.q = count
            self.llm.pos_norm = self.llm.pos_norm[:count]
            for name in ("prom", "payments", "utilities", "ad_len"):
                setattr(self.llm, name, getattr(self.llm, name)[:count])
        request = {
            name: deepcopy(getattr(self.llm, name))
            for name in (
                "query", "basic_reply", "ad_name", "content", "b", "pctr",
                "pos_norm", "q", "max_len", "constant", "sigma", "prom",
                "payments", "utilities", "SW", "ad_len", "dataset", "language",
                "length_unit", "length_counting_rule", "basic_reply_provenance"
            )
        }
        if auction_style == "aal":
            for name in ("aal_retrieval", "aal_noise"):
                request[name] = deepcopy(getattr(self.llm, name))
        if hasattr(self, "pa_aigc_state"):
            request["pa_aigc_auction"] = deepcopy(self.llm.pa_aigc_auction)
        if hasattr(self, "lera_state"):
            for name in ("lera_auction", "lera_retrieval"):
                request[name] = deepcopy(getattr(self.llm, name))
        return request

    def run(self, shared_request=None, *, generation_style="normal"):
        if self.llm.auc_alg is GA:
            return self.run_ga(shared_request)
        if shared_request is None:
            shared_request = self.prepare_request()
        for name, value in shared_request.items():
            setattr(self.llm, name, deepcopy(value))
        self.basic_reply = self.llm.basic_reply
        self.result = {}
        self.result["generation_config"] = deepcopy(self.generation_config)
        self.result["query"] = self.llm.query
        self.result["basic_reply"] = self.basic_reply
        self.result["basic_reply_provenance"] = deepcopy(self.llm.basic_reply_provenance)
        self.result["ad_bids"] = [float(bid) for bid in self.llm.b]
        self.result["ad_pctr"] = [float(ctr) for ctr in self.llm.pctr]
        self.result["ad_rank"] = self.llm.sigma[:self.llm.q].tolist()
        self.result["ad_prom"] = self.llm.prom[:self.llm.q].tolist()
        self.result["ad_sw"] = self.llm.SW
        self.result["ad_rev"] = self.llm.payments.tolist()
        self.result["ad_len"] = self.llm.ad_len
        self.result["ad_utility"] = self.llm.utilities.tolist()
        if hasattr(self.llm, "pa_aigc_auction"):
            self.result["auction"] = deepcopy(self.llm.pa_aigc_auction)
            winners = self.llm.sigma[:self.llm.q]
            values = np.asarray(self.llm.b)[winners]
            reported = values * self.llm.prom[:self.llm.q] - self.llm.payments
            self.result["ad_utility"] = reported.tolist()
            auction = self.result["auction"]
            auction["native_utilities"] = self.llm.utilities.tolist()
            auction["native_per_advertiser_utilities"] = auction["per_advertiser_utilities"]
            auction["native_utility_basis"] = auction["utility_basis"]
            auction["native_value_assumption"] = auction["value_assumption"]
            ledger = np.zeros(len(self.llm.b))
            ledger[winners] = reported
            auction["per_advertiser_utilities"] = ledger.tolist()
            auction["utility_basis"] = "value_times_target_share_minus_expected_payment"
            auction["value_assumption"] = "supplied_bids_equal_values_for_utility_reporting"
        if hasattr(self.llm, "lera_auction"):
            self.result["auction"] = deepcopy(self.llm.lera_auction)
            self.result["retrieval"] = deepcopy(self.llm.lera_retrieval)
            self._report_target_utilities()
        if self.llm.auc_alg is GSP:
            winners = self.llm.sigma[:self.llm.q]
            slot_ctr = np.asarray(self.llm.pctr)[winners] * self.llm.pos_norm[:self.llm.q]
            values = np.asarray(self.llm.b)[winners]
            self.result["ad_utility"] = (values * self.llm.prom[:self.llm.q] - self.llm.payments).tolist()
            self.result["auction"] = {
                "mechanism": "quality_weighted_GSP", "reserve_per_click": 0.0,
                "per_click_prices": (self.llm.payments / slot_ctr).tolist(),
                "position_adjusted_pctr": slot_ctr.tolist(),
                "payment_basis": "expected_per_impression_slot_CTR_times_CPC",
                "utility_basis": "value_times_target_share_minus_expected_payment",
                "native_utilities": self.llm.utilities.tolist(),
                "native_utility_basis": "slot_CTR_times_value_minus_expected_payment",
                "value_assumption": "supplied_bids_equal_values_for_utility_reporting",
                "native_welfare": float(np.dot(np.asarray(self.llm.b)[winners], slot_ctr)),
                "ad_prom_semantics": "equal_rendering_budget_not_CTR",
                "reported_SW": "common_log_platform_objective_at_GSP_allocation"
            }
        for key in ("dataset", "language", "length_unit", "length_counting_rule"):
            self.result[key] = getattr(self.llm, key)

        # Reset context for independent samples.
        base_messages = [
            {"role": "user", "content": self.llm.query},
            {"role": "assistant", "content": self.basic_reply}
        ]
        samples = []
        best_sample = None
        best_messages = None
        for r in range(self.n_samples):
            self.llm.messages = deepcopy(base_messages)
            call_start = len(self.llm.llm_calls)
            self.llm.gen_reply = self.llm.generate_ads(generation_style)
            self.llm.satisfaction()
            sample = {
                "model_calls": deepcopy(self.llm.llm_calls[call_start:]),
                "sample_index": r,
                "gen_reply": self.llm.gen_reply,
                "user_satis": float(self.llm.user_satis),
                "ad_satis": float(self.llm.ad_satis),
                "score": float(self.llm.score),
                "raw_gen_reply": self.llm.raw_gen_reply,
                "ad_spans": deepcopy(self.llm.ad_spans),
                "span_diagnostics": list(self.llm.span_diagnostics),
                "scoring": deepcopy(self.llm.scoring)
            }
            if generation_style == "dwls":
                sample["als_summaries"] = deepcopy(self.llm.als_summaries)
            for key in ("dataset", "language", "length_unit", "length_counting_rule"):
                sample[key] = getattr(self.llm, key)
            if not all(isfinite(sample[key]) for key in ("user_satis", "ad_satis", "score")):
                raise ValueError(f"Non-finite semantic metric or score in sample {r}")
            samples.append(sample)
            if best_sample is None or sample["score"] > best_sample["score"]:
                best_sample = sample
                best_messages = deepcopy(self.llm.messages)

        self.result["samples"] = samples
        self.result["n_samples"] = self.n_samples
        self.result["score_aggregation"] = "mean"
        for key in ("user_satis", "ad_satis", "score"):
            self.result[key] = fsum(sample[key] for sample in samples) / self.n_samples
        self.result["mean_score"] = self.result["score"]
        self.result["best_sample_index"] = best_sample["sample_index"]
        self.result["best_sample_score"] = best_sample["score"]
        self.gen_reply = best_sample["gen_reply"]
        self.result["gen_reply"] = self.gen_reply
        self.llm.gen_reply = self.gen_reply
        self.llm.compressed_ad = self.gen_reply
        self.llm.messages = best_messages
        for key in ("raw_gen_reply", "ad_spans", "span_diagnostics", "scoring"):
            setattr(self.llm, key, deepcopy(best_sample[key]))
        if generation_style == "dwls":
            self.llm.als_summaries = deepcopy(best_sample["als_summaries"])
            self.result["als_summaries"] = deepcopy(best_sample["als_summaries"])
        self.llm._span_reply = self.gen_reply
        self.result["model_calls"] = deepcopy(best_sample["model_calls"])
        self.result["scoring"] = deepcopy(best_sample["scoring"])
        for key in ("user_satis", "ad_satis", "score"):
            setattr(self.llm, key, best_sample[key])
        stamp(self.result, self.llm)
        return self.gen_reply

    def _record_generated_metadata(self):
        for key in ("dataset", "language", "length_unit", "length_counting_rule",
                    "raw_gen_reply", "ad_spans", "span_diagnostics", "scoring"):
            self.result[key] = deepcopy(getattr(self.llm, key))

    def run_DSIC(self):
        call_start = len(self.llm.llm_calls)
        self.basic_reply = self.llm.reply(self.llm.organic_prompt())
        self.llm.basic_reply = self.basic_reply
        self.result["query"] = self.llm.query
        self.result["basic_reply"] = self.basic_reply
        for key in ("dataset", "language", "length_unit", "length_counting_rule"):
            self.result[key] = getattr(self.llm, key)

        self.result["basic_reply_provenance"] = deepcopy(self.llm.llm_calls[call_start:])

        self.llm.calculate_similar(self.basic_reply)

        # Track advertiser identity across rank changes.
        betas = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8]
        original_rank = sorted(
            range(len(self.llm.b)),
            key=lambda i: -(self.llm.b[i] * self.llm.pctr[i])
        )
        self.result["dsic_true_values"] = [float(v) for v in self.llm.b]
        self.result["dsic_pctr"] = [float(p) for p in self.llm.pctr]
        self.result["dsic_betas"] = betas
        self.result["dsic_target_indices"] = original_rank[:self.llm.q]
        for s, target in enumerate(original_rank[:self.llm.q]):
            for beta in betas:
                self.llm.auction_DSIC(beta, s)
                self.result[f"ad_rank_{beta}_{s}"] = self.llm.sigma[:self.llm.q].tolist()
                self.result[f"ad_prom_{beta}_{s}"] = self.llm.prom[:self.llm.q].tolist()
                self.result[f"ad_sw_{beta}_{s}"] = self.llm.SW
                self.result[f"ad_rev_{beta}_{s}"] = self.llm.payments.tolist()
                self.result[f"ad_len_{beta}_{s}"] = self.llm.ad_len
                self.result[f"ad_utility_{beta}_{s}"] = self.llm.utilities.tolist()
                winners = self.llm.sigma[:self.llm.q].tolist()
                target_utility = (float(self.llm.utilities[winners.index(target)])
                                  if target in winners else 0.0)
                self.result[f"ad_target_utility_{beta}_{s}"] = target_utility


    def run_sqa(self, shared_request=None, candidate=None):
        from utils.sqa_protocol import use_fixed_answer
        return use_fixed_answer(self, shared_request, candidate)

    def run_als(self, shared_request=None):
        reply = self.run(shared_request, generation_style="dwls")
        indices = self.llm.sigma[:self.llm.q]
        final_ctr = (np.asarray(self.llm.pctr)[indices] * self.llm.pos_norm[:self.llm.q]
                     * self.llm.prom ** self.als_beta)
        self.result["auction"] = {
            "mechanism": "ALS_DWLS", "beta": float(self.als_beta),
            "alpha": float(1 / (1 - self.als_beta)),
            "payment": "VCG_externality_equivalent_to_Myerson_final_CTR",
            "utility_basis": "value_times_final_CTR_minus_expected_payment",
            "final_pctr": final_ctr.tolist(),
            "native_welfare": float(np.dot(np.asarray(self.llm.b)[indices], final_ctr)),
            "reported_SW": "common_log_platform_objective_at_DWLS_shares",
            "generation": "independent_per_ad_summary_then_ranked_concatenation"
        }
        self._report_target_utilities()
        return reply

    def run_aal(self, shared_request=None):
        if shared_request is None:
            shared_request = self.prepare_request(auction_style="aal")
        reply = self.run(shared_request)
        from auction.AAL import perturbed_ranking
        _, _, log_scores = perturbed_ranking(
            self.llm.b, self.llm.aal_retrieval["relevance"], self.llm.q, self.llm.aal_noise)
        winners = self.llm.sigma[:self.llm.q]
        ctr = np.asarray(self.llm.pctr)[winners]
        bids = np.asarray(self.llm.b)[winners]
        self.result["retrieval"] = deepcopy(self.llm.aal_retrieval)
        self.result["retrieval"]["selected_documents"] = [
            {"advertiser_index": int(i), "ad_name": self.llm.ad_name[i],
             "text": self.llm.content[i]} for i in winners]
        self.result["auction"] = {
            "mechanism": "AAL_multi_allocation_segment", "segments": 1,
            "gumbel_noise": self.llm.aal_noise.tolist(),
            "perturbed_log_scores": [float(x) if np.isfinite(x) else None for x in log_scores],
            "critical_per_click_prices": (self.llm.payments / ctr).tolist(),
            "payment_basis": "conditional_on_random_draw_expected_over_clicks",
            "utility_basis": "base_CTR_times_(value_minus_critical_per_click_price)",
            "native_welfare": float(np.dot(bids, ctr)),
            "ad_prom_semantics": "equal_rendering_budget_not_retrieval_probability",
            "reported_SW": "common_log_platform_objective_at_rendered_winner_set",
            "calibrated_retriever_assumption": "semantic_relevance_and_shared_synthetic_CTRs"
        }
        self._report_target_utilities()
        return reply

    def _report_target_utilities(self):
        result = self.result
        winners = np.asarray(result["ad_rank"], dtype=int)
        native = np.asarray(result["ad_utility"], dtype=float)
        values = np.asarray(result["ad_bids"])[winners]
        reported = values * result["ad_prom"] - result["ad_rev"]
        native_ledger, ledger = (np.zeros(len(result["ad_bids"])) for _ in range(2))
        native_ledger[winners], ledger[winners] = native, reported
        auction = result["auction"]
        auction["native_utilities"] = native.tolist()
        auction["native_per_advertiser_utilities"] = native_ledger.tolist()
        auction["native_utility_basis"] = auction["utility_basis"]
        if "value_assumption" in auction:
            auction["native_value_assumption"] = auction["value_assumption"]
        auction["per_advertiser_utilities"] = ledger.tolist()
        auction["utility_basis"] = "value_times_target_share_minus_expected_payment"
        auction["value_assumption"] = "supplied_bids_equal_values_for_utility_reporting"
        result["ad_utility"] = reported.tolist()

    def run_ga(self, shared_request=None):
        if shared_request is None:
            shared_request = self.prepare_request()
        for name, value in shared_request.items():
            setattr(self.llm, name, deepcopy(value))
        self.basic_reply = self.llm.basic_reply
        count = int(np.count_nonzero(self.llm.prom[:self.llm.q] > 0))
        ids = self.llm.sigma[:count]
        paid, utility = np.zeros(len(self.llm.b)), np.zeros(len(self.llm.b))
        paid[ids] = self.llm.payments[:count]
        utility[ids] = self.llm.utilities[:count]
        self.gen_reply = self.llm.generate_ads("ga")
        self.llm.gen_reply = self.gen_reply
        self.llm.satisfaction()
        self.result = {
            "generation_config": deepcopy(self.generation_config),
            "query": self.llm.query, "basic_reply": self.basic_reply,
            "basic_reply_provenance": deepcopy(self.llm.basic_reply_provenance), "model_calls": [],
            "ad_bids": np.asarray(self.llm.b).tolist(), "ad_pctr": np.asarray(self.llm.pctr).tolist(),
            "ad_rank": ids.tolist(), "ad_prom": self.llm.prom[:count].tolist(),
            "ad_rev": self.llm.payments[:count].tolist(), "ad_utility": self.llm.utilities[:count].tolist(),
            "ad_len": list(self.llm.ad_len[:count]), "ad_sw": self.llm.SW, "gen_reply": self.gen_reply,
            "auction": {
                "mechanism": "GA_ranked_length_greedy", "configured_capacity": self.llm.q,
                "displayed_ad_count": count,
                "score_order": np.argsort(-np.asarray(self.llm.b)*self.llm.pctr).tolist(),
                "allocation": "ranked_remaining_budget_source_prefix",
                "payment_rule_status": "legacy_formula",
                "utility_basis": "legacy_value_times_text_share_minus_payment",
                "per_advertiser_payments": paid.tolist(), "per_advertiser_utilities": utility.tolist(),
                "reported_SW": "common_log_platform_objective_over_displayed_ads_only"
            }
        }
        self._record_generated_metadata()
        for key in ("user_satis", "ad_satis", "score"):
            self.result[key] = float(getattr(self.llm, key))
        sample = {key: deepcopy(self.result[key]) for key in (
            "gen_reply", "user_satis", "ad_satis", "score", "raw_gen_reply", "ad_spans", "model_calls",
            "span_diagnostics", "scoring", "dataset", "language", "length_unit", "length_counting_rule")}
        sample["sample_index"] = 0
        self.result.update(samples=[sample], n_samples=1, score_aggregation="deterministic",
                           mean_score=self.result["score"], best_sample_index=0,
                           best_sample_score=self.result["score"])
        stamp(self.result, self.llm)
        return self.gen_reply
