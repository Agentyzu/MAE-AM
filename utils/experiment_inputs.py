from copy import deepcopy
import hashlib
import json

import numpy as np


def input_hash(instance):
    fields = ("instance_id", "repeat_index", "seed", "config", "advertisers",
              "bids", "true_values", "basic_reply", "base_model", "source")
    payload = {key: instance[key] for key in fields}
    if "basic_reply_provenance" in instance:
        payload["basic_reply_provenance"] = instance["basic_reply_provenance"]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def validate(instance, *, frozen=True):
    if instance.get("input_hash") != input_hash(instance):
        raise ValueError("Paired input hash mismatch; prepare a new manifest for changed inputs")
    data, cfg = instance["advertisers"], instance["config"]
    n = len(data)
    bids, values = np.asarray(instance["bids"]), np.asarray(instance["true_values"])
    ctr = np.asarray([row["pctr"] for row in data], dtype=float)
    if (n < cfg["q"] or n < 2 or bids.shape != (n,) or values.shape != (n,)
            or not np.all(np.isfinite(bids)) or np.any(bids < 0)
            or not np.array_equal(bids, values)):
        raise ValueError("Paired default runs require n>=max(q,2) and finite truthful bid/value vectors")
    if not np.all(np.isfinite(ctr)) or np.any(ctr < .3) or np.any(ctr > .5):
        raise ValueError("Paired CTRs must lie in the paper's [0.3,0.5] range")
    if not cfg["models"] or instance["base_model"] != cfg["models"][0]:
        raise ValueError("The shared base model must be the first candidate model")
    if instance["source"].get("pool_policy") == "category_ten_length_intervals_without_replacement":
        from utils.candidate_pools import identity
        from utils.text_length import count_text
        source = instance["source"]
        if n != cfg["n_advertisers"] or n != source["requested_n"] or source["interval_count"] != 10:
            raise ValueError("Stratified pool does not have the declared candidate/interval counts")
        names = [identity(row["ad_name"]) for row in data]
        texts = [identity(row.get("content", row.get("ad_copy"))) for row in data]
        ids = [row["source_record_id"] for row in data]
        if len(set(names)) != n or len(set(texts)) != n or len(set(ids)) != n:
            raise ValueError("Candidate pool contains repeated advertisers, creatives or source rows")
        measured = [count_text(row.get("content", row.get("ad_copy")), source["length_unit"]) for row in data]
        bins = np.clip(np.searchsorted(source["bin_edges"], measured, side="right") - 1, 0, 9)
        if (any(row["source_length"] != size or row["length_interval"] != int(bin_id)
                for row, size, bin_id in zip(data, measured, bins))
                or np.bincount(bins, minlength=10).tolist() != source["sampled_bin_counts"]):
            raise ValueError("Candidate length intervals do not match recorded source text lengths")
    if frozen and not isinstance(instance["basic_reply"], str):
        raise ValueError("Freeze the common organic reply before binding an instance")


def bind(zoo, instance):
    validate(instance)
    cfg = instance["config"]
    if list(zoo.llms) != cfg["models"]:
        raise ValueError("Paired candidate models must match the manifest in order")
    for key in ("query", "q", "max_len", "constant", "theta_1", "theta_2", "is_structured"):
        if zoo.cfg[key] != cfg[key]:
            raise ValueError(f"Paired configuration mismatch: {key}")
    if zoo.cfg.get("n_samples", 5) != cfg["n_samples"]:
        raise ValueError("Paired sampling budget mismatch")
    if zoo.cfg.get("backbones", {}) != cfg.get("backbones", {}):
        raise ValueError("Paired backbone configuration mismatch")
    for process in zoo.llms.values():
        model = process.llm
        expected = instance["advertisers"]
        texts = [row.get("content", row.get("ad_copy")) for row in expected]
        if model.content != texts or model.ad_name != [row["ad_name"] for row in expected]:
            raise ValueError("Paired advertiser identity/order does not match the frozen pool")
        if model.dataset != cfg["dataset"]:
            raise ValueError("Paired dataset mismatch")
        model.shared_instance = deepcopy(instance)
    return zoo


def apply(model, *, include_bids=True):
    instance = getattr(model, "shared_instance", None)
    if instance is None:
        return False
    validate(instance)
    cfg = instance["config"]
    for key in ("query", "q", "max_len", "constant", "theta_1", "theta_2", "is_structured"):
        setattr(model, key, deepcopy(cfg[key]))
    model.basic_reply = instance["basic_reply"]
    model.basic_reply_provenance = deepcopy(instance.get("basic_reply_provenance", []))
    model.pctr = np.array([row["pctr"] for row in instance["advertisers"]], dtype=float)
    model.pos_norm = .9 ** np.arange(1, cfg["q"] + 1)
    if include_bids:
        model.b = np.array(instance["bids"], dtype=float)
    return True


def stamp(result, model):
    instance = getattr(model, "shared_instance", None)
    if instance is not None:
        result["experiment"] = {
            "instance_id": instance["instance_id"], "repeat_index": instance["repeat_index"],
            "input_hash": instance["input_hash"], "seed": instance["seed"],
            "base_model": instance["base_model"], "actual_advertiser_count": len(instance["advertisers"]),
            "pool_policy": instance["source"].get("pool_policy", "existing_group_preserved_no_cross_interval_resampling")
        }
    return result
