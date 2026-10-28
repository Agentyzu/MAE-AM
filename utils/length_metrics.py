from collections import Counter, defaultdict
from math import fsum, isfinite, sqrt
from numbers import Real

from utils.text_length import RULES, count_text


UNITS = {"ADGEN": "characters", "ATVI": "words"}


def _count(value):
    return (isinstance(value, Real) and not isinstance(value, bool)
            and isfinite(value) and value >= 0 and value == int(value))


def summarize_pairs(pairs):
    pairs = list(pairs)
    if any(not _count(x) or not _count(y) for x, y in pairs):
        raise ValueError("Length pairs must contain finite nonnegative integer counts")
    n = len(pairs)
    target = sum(int(x) for x, _ in pairs)
    generated = sum(int(y) for _, y in pairs)
    error = generated - target
    absolute = sum(abs(int(y) - int(x)) for x, y in pairs)
    pearson, reason = None, "fewer_than_two_pairs"
    if n >= 2:
        mx, my = target / n, generated / n
        xx = fsum((x - mx) ** 2 for x, _ in pairs)
        yy = fsum((y - my) ** 2 for _, y in pairs)
        if xx == 0 or yy == 0:
            reason = "constant_target_or_generated_lengths"
        else:
            xy = fsum((x - mx) * (y - my) for x, y in pairs)
            pearson = max(-1., min(1., xy / sqrt(xx) / sqrt(yy)))
            reason = None
    nmae = absolute / target if target else None
    return {
        "paired_ad_count": n, "target_length_total": target,
        "generated_length_total": generated, "signed_error_total": error,
        "absolute_error_total": absolute,
        "Bias": error / n if n else None, "MAE": absolute / n if n else None,
        "pooled_NMAE": nmae, "pooled_NMAE_percent": 100 * nmae if nmae is not None else None,
        "Pearson": pearson, "Pearson_undefined_reason": reason,
        "NMAE_undefined_reason": "zero_target_total" if not target else None,
    }


def _record_pairs(record):
    ranks, lengths, spans = (record.get(k) for k in ("ad_rank", "ad_len", "ad_spans"))
    if not isinstance(ranks, list) or not isinstance(lengths, list) or not isinstance(spans, list):
        return None, Counter(), "missing_advertiser_length_records"
    if (len(ranks) != len(lengths) or any(not _count(i) for i in ranks)
            or len(set(ranks)) != len(ranks)):
        return None, Counter(), "invalid_advertiser_order_or_targets"
    by_id = defaultdict(list)
    excluded = Counter()
    for span in spans:
        if not isinstance(span, dict) or not _count(span.get("advertiser_index")):
            excluded["invalid_span_record"] += 1
        elif span["advertiser_index"] not in ranks:
            excluded["unselected_advertiser_span"] += 1
        else:
            by_id[span["advertiser_index"]].append(span)
    pairs = []
    for i, expected in zip(ranks, lengths):
        records = by_id[i]
        if len(records) != 1:
            excluded["missing_span" if not records else "duplicate_span"] += 1
            continue
        span = records[0]
        status = span.get("status")
        if status not in ("ok", "empty"):
            excluded[status if status in ("missing", "invalid") else "unrecognized_span_status"] += 1
            continue
        x, y = span.get("target_length"), span.get("generated_length")
        if span.get("length_unit") != record["length_unit"]:
            raise ValueError("Span and reply length units disagree")
        if not _count(x) or not _count(y) or not _count(expected) or x != expected:
            excluded["invalid_or_inconsistent_length_pair"] += 1
            continue
        text = span.get("text")
        if not isinstance(text, str) or count_text(text, record["length_unit"]) != y:
            excluded["generated_count_text_mismatch"] += 1
            continue
        if (status == "empty") != (not text.strip()):
            excluded["span_status_text_mismatch"] += 1
            continue
        if span.get("length_error") != y - x:
            excluded["signed_error_mismatch"] += 1
            continue
        pairs.append((int(x), int(y)))
    return pairs, excluded, None


def summarize_records(entries):
    groups, skipped = {}, []
    for entry in entries:
        record = entry["record"]
        def skip(reason):
            skipped.append({"source": entry["source"], "scope": entry["scope"], "reason": reason})
        if record.get("validation_only") or record.get("scoring", {}).get("validation_only"):
            skip("validation_record")
            continue
        if "samples" in record:
            index, samples = record.get("best_sample_index"), record["samples"]
            if not isinstance(samples, list) or not _count(index) or index >= len(samples):
                skip("missing_selected_model_sample")
                continue
            record = dict(record, **samples[int(index)])
        if record.get("validation_only") or record.get("scoring", {}).get("validation_only"):
            skip("validation_record")
            continue
        dataset, unit = record.get("dataset"), record.get("length_unit")
        if dataset not in UNITS or unit is None:
            skip("missing_dataset_or_length_unit")
            continue
        if (unit != UNITS[dataset] or entry.get("dataset", dataset) != dataset
                or record.get("length_counting_rule", RULES[unit]) != RULES[unit]):
            raise ValueError("Dataset, length unit or counting rule disagrees with ADGEN/ATVI convention")
        cfg = record.get("generation_config", entry.get("generation_config"))
        if (not isinstance(cfg, dict) or not _count(cfg.get("q")) or cfg["q"] < 1
                or not _count(cfg.get("max_len")) or cfg["max_len"] < 1
                or not isinstance(cfg.get("is_structured"), bool)):
            skip("missing_or_invalid_generation_config")
            continue
        if "generation_config" in entry and cfg != entry["generation_config"]:
            raise ValueError("Recorded generation configuration disagrees with manifest")
        pairs, excluded, reason = _record_pairs(record)
        if reason:
            skip(reason)
            continue
        calls = record.get("model_calls", [])
        requested = tuple(sorted({c["requested_model"] for c in calls if c.get("requested_model")}))
        returned = tuple(sorted({c["returned_model"] for c in calls if c.get("returned_model")}))
        key = (entry["scope"], dataset, unit, entry["method"], entry["model"],
               int(cfg["q"]), int(cfg["max_len"]), cfg["is_structured"], requested, returned)
        if key not in groups:
            groups[key] = dict(pairs=[], excluded=Counter(), replies=0, paired_replies=0,
                               repeats=set(), sources=[])
        group = groups[key]
        group["pairs"].extend(pairs)
        group["excluded"].update(excluded)
        group["replies"] += 1
        group["paired_replies"] += bool(pairs)
        group["sources"].append(entry["source"])
        if entry.get("repeat_index") is not None:
            group["repeats"].add(entry["repeat_index"])
    result = []
    for key, group in sorted(groups.items()):
        scope, dataset, unit, method, model, q, budget, structured, requested, returned = key
        result.append(dict(
            scope=scope, dataset=dataset, length_unit=unit, method=method, model=model,
            q=q, max_len=budget, is_structured=structured,
            requested_models=list(requested), returned_models=list(returned),
            reply_count=group["replies"], paired_reply_count=group["paired_replies"],
            repeat_indices=sorted(group["repeats"]), sources=group["sources"],
            excluded_ad_records=dict(sorted(group["excluded"].items())),
            **summarize_pairs(group["pairs"])))
    return dict(
        aggregation="pool_advertisement_pairs_within_dataset_method_model_configuration_and_scope",
        scopes={"per_model_best_sample": "one selected sample per candidate model per instance",
                "selected_reply": "one delivered reply per instance; kept separate from per-model samples"},
        groups=result, skipped_records=skipped,
    )
