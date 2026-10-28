import argparse
import ast
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from math import fsum, isclose, isfinite
from numbers import Real
import os
from pathlib import Path
import random

import numpy as np

from utils.experiment_inputs import input_hash, validate

CODE = Path(__file__).resolve().parent
METHODS = {"Prom": "Prom", "GSP": "GSP", "GA": "GA",
           "ALS": "Prom_ALS", "AAL": "Prom_AAL", "SQA": "GSP_SQA", "PA-AIGC": "PA_AIGC", "LERA": "LERA"}
METRICS = ("SW", "Rev", "Util", "Su", "Sa", "Score")
PAPER_MODELS = ("ChatGPT", "claude", "kimi", "GLM", "ernie", "Qwen")


class Config(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def derived_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:4], "big")


@contextmanager
def local_randomness(seed):
    numpy_state, python_state = np.random.get_state(), random.getstate()
    np.random.seed(seed)
    random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def read_config(dataset, category):
    path = CODE / f"cfgs/{dataset}_Prom_configs/config_{category}_Prom.py"
    values = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                    and target.value.id == "__C"):
                values[target.attr] = ast.literal_eval(node.value)
    values.update(dataset=dataset, n_samples=5)
    values.pop("auc", None)
    return values


def experiment_config(dataset, category, *, models=None, profile="legacy"):
    cfg = read_config(dataset, category)
    if profile == "paper":
        if models is not None and (len(models) != len(PAPER_MODELS) or set(models) != set(PAPER_MODELS)):
            raise ValueError("The paper profile requires all six PAPER_MODELS; use legacy for a subset")
        cfg.update(models=list(PAPER_MODELS), is_structured=True)
    elif profile != "legacy":
        raise ValueError("Choose the legacy or paper experiment profile")
    if models is not None:
        cfg["models"] = list(models)
    return cfg


def _prepare_existing_manifest(output, *, datasets=("ADGEN", "ATVI"), groups=None, models=None,
                     seed=20261027, repetitions=5, profile="legacy"):
    output = Path(output).resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError("Manifest already exists; run/resume it or choose a new output folder")
    if repetitions < 1 or not datasets or len(set(datasets)) != len(datasets):
        raise ValueError("Use at least one dataset and a positive repetition count")
    if models is not None and (not models or len(set(models)) != len(models)):
        raise ValueError("Candidate models must be a nonempty unique ordered list")
    wanted = set(groups or [])
    found, instances = set(), []
    for dataset in datasets:
        if dataset not in ("ADGEN", "ATVI"):
            raise ValueError(f"Unsupported dataset: {dataset}")
        for path in sorted((CODE / "dataset" / dataset).glob("*/*_section*.json")):
            group = f"{dataset}/{path.stem}"
            if wanted and group not in wanted:
                continue
            found.add(group)
            original = json.loads(path.read_text(encoding="utf-8"))
            cfg = experiment_config(dataset, path.parent.name, models=models, profile=profile)
            for repeat in range(repetitions):
                row_seed = derived_seed(seed, group, repeat)
                rng = np.random.default_rng(row_seed)
                data = deepcopy(original)
                for index, row in enumerate(data):
                    row["source_record_id"] = f"{path.relative_to(CODE).as_posix()}#{index}"
                    row["pctr"] = float(rng.uniform(.3, .5))
                values = rng.uniform(0, 1, len(data)).tolist()
                instance = {
                    "instance_id": f"{dataset}_{path.stem}_r{repeat + 1}",
                    "repeat_index": repeat, "seed": row_seed, "config": deepcopy(cfg),
                    "advertisers": data, "true_values": values, "bids": list(values),
                    "basic_reply": None, "base_model": cfg["models"][0],
                    "source": {"group": group, "path": path.relative_to(CODE).as_posix(),
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                }
                instance["input_hash"] = input_hash(instance)
                validate(instance, frozen=False)
                instances.append(instance)
    if wanted - found:
        raise ValueError(f"Unknown groups: {sorted(wanted - found)}")
    if not instances:
        raise ValueError("No supplied dataset groups matched")
    manifest = {"schema_version": 1, "seed": seed, "repetitions": repetitions,
                "pool_policy": "existing_group_preserved_no_cross_interval_resampling",
                "implemented_methods": list(METHODS), "not_implemented": [],
                "instances": instances}
    if profile == "paper":
        manifest["profile"] = profile
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "manifest.json", manifest)
    return output / "manifest.json"



def prepare_manifest(output, *, datasets=("ADGEN", "ATVI"), groups=None, models=None,
                     seed=20261027, repetitions=5, categories=None, pool_policy=None,
                     n_advertisers=20, profile="legacy"):
    from utils.candidate_pools import POLICY, load_pools, sample_pool
    policy = pool_policy or ("existing_groups" if groups else "stratified_category")
    if policy == "existing_groups":
        if categories or n_advertisers != 20:
            raise ValueError("Category/n options require the stratified_category policy")
        return _prepare_existing_manifest(output, datasets=datasets, groups=groups,
                                          models=models, seed=seed, repetitions=repetitions, profile=profile)
    if policy != "stratified_category" or groups:
        raise ValueError("Use --categories for stratified_category or --groups for existing_groups")
    output = Path(output).resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError("Manifest already exists; run/resume it or choose a new output folder")
    if (not datasets or len(set(datasets)) != len(datasets)
            or any(dataset not in ("ADGEN", "ATVI") for dataset in datasets)
            or isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1):
        raise ValueError("Use supported unique datasets and a positive integer repetition count")
    if models is not None and (not models or len(set(models)) != len(models)):
        raise ValueError("Candidate models must be a nonempty unique ordered list")
    pools, excluded = load_pools(CODE, datasets)
    wanted = set(categories or [])
    known = {pool["source"]["group"] for pool in pools}
    if wanted - known:
        raise ValueError(f"Unknown configured categories: {sorted(wanted - known)}")
    instances, inventory = [], []
    for pool in pools:
        group = pool["source"]["group"]
        if wanted and group not in wanted:
            continue
        cfg = experiment_config(pool["dataset"], pool["category"], models=models, profile=profile)
        if (isinstance(n_advertisers, bool) or not isinstance(n_advertisers, int)
                or n_advertisers < max(2, cfg["q"])):
            raise ValueError("n_advertisers must be an integer at least max(q,2)")
        inventory.append(deepcopy(pool["source"]))
        if pool["source"]["distinct_advertisers"] < n_advertisers:
            reason = {**deepcopy(pool["source"]), "reason": "insufficient_distinct_advertisers",
                      "requested_n": n_advertisers}
            if wanted:
                raise ValueError(f"Requested {group} has {pool['source']['distinct_advertisers']} distinct advertisers; need {n_advertisers}")
            excluded.append(reason)
            continue
        cfg["n_advertisers"] = n_advertisers
        for repeat in range(repetitions):
            row_seed = derived_seed(seed, group, repeat)
            data, source = sample_pool(pool, np.random.default_rng(derived_seed(row_seed, "pool")), n=n_advertisers)
            rng = np.random.default_rng(row_seed)
            for row in data:
                row["pctr"] = float(rng.uniform(.3, .5))
            values = rng.uniform(0, 1, n_advertisers).tolist()
            instance = {
                "instance_id": f"{pool['dataset']}_{pool['category']}_r{repeat + 1}",
                "repeat_index": repeat, "seed": row_seed, "config": deepcopy(cfg),
                "advertisers": data, "true_values": values, "bids": list(values),
                "basic_reply": None, "base_model": cfg["models"][0], "source": source}
            instance["input_hash"] = input_hash(instance)
            validate(instance, frozen=False)
            instances.append(instance)
    if not instances:
        raise ValueError("No category can supply the requested distinct advertisers; no ads were duplicated")
    manifest = {"schema_version": 1, "seed": seed, "repetitions": repetitions,
                "pool_policy": POLICY, "n_advertisers": n_advertisers, "length_intervals": 10,
                "category_inventory": inventory, "excluded_categories": excluded,
                "implemented_methods": list(METHODS), "not_implemented": [],
                "instances": instances}
    if profile == "paper":
        manifest["profile"] = profile
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "manifest.json", manifest)
    return output / "manifest.json"


def registry():
    from model_zoo.model_zoo import ModelZoo
    from model_zoo.model_zoo_GA import ModelZoo_GA
    from model_zoo.model_zoo_ALS import ModelZoo_ALS
    from model_zoo.model_zoo_AAL import ModelZoo_AAL
    from model_zoo.model_zoo_SQA import ModelZoo_SQA
    return {"Prom": ModelZoo, "GSP": ModelZoo, "GA": ModelZoo_GA,
            "ALS": ModelZoo_ALS, "AAL": ModelZoo_AAL, "SQA": ModelZoo_SQA, "PA-AIGC": ModelZoo, "LERA": ModelZoo}


def checked_methods(methods):
    if not methods or len(set(methods)) != len(methods) or any(x not in METHODS for x in methods):
        raise ValueError("Choose unique implemented methods from: " + ", ".join(METHODS))
    return list(methods)


def selected_path(root, instance, method):
    name = instance["instance_id"]
    return Path(root) / f"result_{METHODS[method]}" / name / f"selected_results_{name}.json"


def assert_paired(record, instance):
    evidence = record.get("experiment", {})
    if (evidence.get("input_hash") != instance["input_hash"]
            or evidence.get("instance_id") != instance["instance_id"]
            or evidence.get("repeat_index") != instance["repeat_index"]):
        raise ValueError("Result does not belong to this frozen paired input")
    if (record["ad_bids"] != instance["bids"]
            or record["ad_pctr"] != [row["pctr"] for row in instance["advertisers"]]
            or record["basic_reply"] != instance["basic_reply"]
            or record["base_model"] != instance["base_model"]
            or record["query"] != instance["config"]["query"]
            or record["candidate_models"] != instance["config"]["models"]):
        raise ValueError("Method resampled or changed paired common inputs")

    if "basic_reply_provenance" in instance:
        if record.get("basic_reply_provenance") != instance["basic_reply_provenance"]:
            raise ValueError("Common organic model provenance changed")
    if record.get("auction", {}).get("mechanism") == "LERA_two_stage_critical_CPC":
        if record.get("retrieval", {}).get("service", {}).get("settings") != instance["config"].get("lera"):
            raise ValueError("LERA scoring service differs from the frozen configuration")
    for call in record.get("model_calls", []):
        entry = instance["config"].get("backbones", {}).get(call["backend"])
        if entry and (call["requested_model"] != entry["model_id"]
                      or call["base_url"] != entry["base_url"]
                      or call["parameters"] != entry["parameters"]):
            raise ValueError("Generated reply used a different backbone configuration")


def assert_complete(record, instance, method):
    from utils.ad_spans import extract_ad_spans
    from utils.text_length import count_text

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def finite(value):
        return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)

    try:
        require(isinstance(record, dict), "result must be an object")
        assert_paired(record, instance)
        cfg = instance["config"]
        rank = record["ad_rank"]
        require(isinstance(rank, list) and len(rank) <= cfg["q"]
                and all(type(i) is int and 0 <= i < len(instance["advertisers"]) for i in rank)
                and len(set(rank)) == len(rank), "invalid selected advertiser indices")
        for key in ("ad_prom", "ad_rev", "ad_utility", "ad_len"):
            require(isinstance(record[key], list) and len(record[key]) == len(rank)
                    and all(finite(v) for v in record[key]), f"invalid {key}")
        if method == "GA":
            require(record["auction"]["mechanism"] == "GA_ranked_length_greedy"
                    and record["auction"]["allocation"] == "ranked_remaining_budget_source_prefix",
                    "GA result uses a different allocation rule")
        if method in ("GSP", "PA-AIGC", "AAL", "ALS", "SQA", "LERA"):
            require(record["auction"]["utility_basis"] == "value_times_target_share_minus_expected_payment",
                    f"{method} result uses a different utility definition")
            for k, i in enumerate(rank):
                expected = instance["true_values"][i] * record["ad_prom"][k] - record["ad_rev"][k]
                require(isclose(record["ad_utility"][k], expected, rel_tol=1e-9, abs_tol=1e-9),
                        f"{method} target-share utility mismatch")
        for key in ("ad_sw", "user_satis", "ad_satis", "score", "selected_model_mean_score"):
            require(finite(record[key]), f"invalid {key}")
        require(all(isinstance(record[key], str) for key in ("gen_reply", "raw_gen_reply")),
                "missing reply text")
        require(isinstance(record["model_calls"], list), "invalid model_calls")
        for call in record["model_calls"]:
            require(isinstance(call, dict) and all(key in call for key in (
                "backend", "requested_model", "base_url", "parameters", "called_at_utc",
                "returned_model", "response_id", "system_fingerprint", "finish_reason", "sdk", "sdk_version")),
                "incomplete model call record")
        unit = "characters" if cfg["dataset"] == "ADGEN" else "words"
        require(record["dataset"] == cfg["dataset"] and record["length_unit"] == unit,
                "dataset/length unit mismatch")
        clean, spans, diagnostics = extract_ad_spans(record["raw_gen_reply"], rank)
        if method == "ALS":
            # Restore attribution from per-ad raw replies.
            summaries = record["als_summaries"]
            require(isinstance(summaries, list) and len(summaries) == len(rank),
                    "incomplete ALS summary records")
            for k, (summary, span) in enumerate(zip(summaries, spans)):
                require(summary["advertiser_index"] == rank[k]
                        and summary["target_length"] == record["ad_len"][k], "ALS summary order/target mismatch")
                _, original, reasons = extract_ad_spans(summary["raw_reply"], [rank[k]])
                source = original[0]
                require(summary["status"] == source["status"] and summary["diagnostics"] == reasons
                        and source["text"] == span["text"], "ALS summary attribution mismatch")
                if source["status"] in ("missing", "invalid"):
                    require(span["status"] == "empty", "ALS failed summary has attributed text")
                    span.update(status=source["status"], start=None, end=None)
                    diagnostics.append(f'als_summary_{source["status"]}:{rank[k]}')
        require(record["gen_reply"] == clean and record["span_diagnostics"] == diagnostics,
                "reply/attribution mismatch")
        require(isinstance(record["ad_spans"], list) and len(record["ad_spans"]) == len(rank),
                "incomplete ad_spans")
        for k, (saved, span) in enumerate(zip(record["ad_spans"], spans)):
            require(isinstance(saved, dict) and all(saved[key] == value for key, value in span.items()),
                    "advertiser span does not match reply")
            if method == "ALS" and span["status"] in ("missing", "invalid"):
                require(saved["cosine"] == 0, "ALS unattributed summary must score zero")
            length = count_text(span["text"], unit) if span["status"] in ("ok", "empty") else None
            target = record["ad_len"][k]
            require(saved["length_unit"] == unit and saved["target_length"] == target
                    and saved["generated_length"] == length
                    and saved["length_error"] == (length - target if length is not None else None)
                    and finite(saved["cosine"]), "incomplete or inconsistent span measurements")
        scoring = record["scoring"]
        require(isinstance(scoring, dict) and all(key in scoring for key in (
            "model", "metric", "Su_scope", "Sa_scope", "ad_span_policy", "missing_or_invalid_span_score"))
                and scoring["Sa_denominator"] == len(rank), "incomplete scoring metadata")
        sa = fsum(span["cosine"] for span in record["ad_spans"]) / len(rank) if rank else 0.0
        score = cfg["theta_1"] * record["user_satis"] + (1 - cfg["theta_1"]) * sa + cfg["theta_2"] * record["ad_sw"]
        require(isclose(record["ad_satis"], sa, rel_tol=1e-9, abs_tol=1e-9)
                and isclose(record["score"], score, rel_tol=1e-9, abs_tol=1e-9), "inconsistent Sa/Score")
        n_samples = 1 if method == "GA" else cfg.get("n_samples", 5)
        index = record["selected_sample_index"]
        require(type(record["n_samples"]) is int and record["n_samples"] == n_samples
                and type(index) is int and 0 <= index < n_samples and record["sample_index"] == index
                and record["score_aggregation"] == "selected_sample", "invalid selected sample metadata")
        means = record["model_mean_scores"]
        require(isinstance(means, dict) and set(means) == set(cfg["models"])
                and all(finite(v) for v in means.values()), "incomplete model mean scores")
        selected = max(cfg["models"], key=means.__getitem__)
        require(record["selected_model"] == selected
                and isclose(record["selected_model_mean_score"], means[selected], rel_tol=1e-9, abs_tol=1e-9),
                "inconsistent selected model")
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Incomplete or inconsistent {method} selected result: {exc}") from exc


def load_completed_result(path, instance, method):
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        assert_complete(record, instance, method)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Cannot reuse or aggregate {path}: {exc}. "
                         "Preserve this file for inspection and rerun in a new output directory "
                         "or explicitly remove the damaged result before resuming.") from exc
    return record


def load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest["instances"]:
        raise ValueError("Unsupported or empty paired manifest")
    ids = [row["instance_id"] for row in manifest["instances"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated instance identifiers in manifest")
    for row in manifest["instances"]:
        if not 0 <= row["repeat_index"] < manifest["repetitions"]:
            raise ValueError("Repeat index lies outside the declared repetition range")
        validate(row, frozen=False)
    return manifest


def run_manifest(path, *, methods=tuple(METHODS)):
    path = Path(path).resolve()
    manifest, methods = load_manifest(path), checked_methods(methods)
    classes = registry()
    root, cwd = path.parent, Path.cwd()
    try:
        os.chdir(root)
        for instance in manifest["instances"]:
            cfg, data = instance["config"], instance["advertisers"]
            if "LERA" in methods and instance["basic_reply"] is not None and "lera" not in cfg:
                raise ValueError("LERA requires frozen scoring-service settings; prepare a new manifest")
            if instance["basic_reply"] is None:
                if "LERA" in methods:
                    from utils.lera_retrieval import settings
                    cfg["lera"] = settings(cfg.get("lera", {}), cfg["q"])
                organic_cfg = Config(deepcopy(cfg), auc="Prom")
                zoo = classes["Prom"](organic_cfg, deepcopy(data), instance["instance_id"])
                # Freeze resolved backends before generation.
                from utils.backbones import settings
                for name, process in zoo.llms.items():
                    if getattr(process.llm, "backend_name", None) == name:
                        spec, _ = settings(process.llm, name)
                        cfg.setdefault("backbones", {})[name] = {
                            "model_id": spec["requested_model"], "base_url": spec["base_url"],
                            "parameters": deepcopy(spec["parameters"])}
                for process in zoo.llms.values():
                    process.llm.backbones = deepcopy(cfg.get("backbones", {}))
                model = zoo.llms[instance["base_model"]].llm
                with local_randomness(derived_seed(instance["seed"], "organic")):
                    instance["basic_reply"] = model.reply(model.organic_prompt())
                instance["basic_reply_provenance"] = deepcopy(model.llm_calls)
                instance["input_hash"] = input_hash(instance)
                validate(instance)
                atomic_json(path, manifest)  # Persist inputs before running auctions.
            for method in methods:
                result_path = selected_path(root, instance, method)
                if result_path.exists():
                    load_completed_result(result_path, instance, method)
                    continue
                method_cfg = Config(deepcopy(cfg), auc=METHODS[method])
                zoo = classes[method](method_cfg, deepcopy(data), instance["instance_id"])
                zoo.bind_instance(instance)
                with local_randomness(derived_seed(instance["seed"], method)):
                    zoo.run_all_models()
                assert_complete(zoo.final_result, instance, method)
        return aggregate(path, methods=methods)
    finally:
        os.chdir(cwd)


def aggregate(path, *, methods=tuple(METHODS)):
    path = Path(path).resolve()
    manifest, methods = load_manifest(path), checked_methods(methods)
    repeats = manifest["repetitions"]
    expected_groups = {row["source"]["group"] for row in manifest["instances"]}
    per_repeat = []
    dataset_repeats = {row["config"]["dataset"]: [] for row in manifest["instances"]}
    for method in methods:
        for repeat in range(repeats):
            rows = [row for row in manifest["instances"] if row["repeat_index"] == repeat]
            if ({row["source"]["group"] for row in rows} != expected_groups
                    or len(rows) != len(expected_groups)):
                raise ValueError("Each repeat must contain exactly the same instance groups")
            metrics = []
            dataset_metrics = {dataset: [] for dataset in dataset_repeats}
            for instance in rows:
                validate(instance)
                record = load_completed_result(selected_path(path.parent, instance, method), instance, method)
                metric = dict(SW=float(record["ad_sw"]), Rev=fsum(record["ad_rev"]),
                              Util=fsum(record["ad_utility"]), Su=float(record["user_satis"]),
                              Sa=float(record["ad_satis"]), Score=float(record["score"]))
                metrics.append(metric)
                dataset_metrics[instance["config"]["dataset"]].append(metric)
            per_repeat.append({"method": method, "repeat_index": repeat, "instances": len(rows),
                               "metrics": {key: fsum(m[key] for m in metrics) / len(metrics) for key in METRICS}})
            for dataset, scoped in dataset_metrics.items():
                dataset_repeats[dataset].append({
                    "method": method, "repeat_index": repeat, "instances": len(scoped),
                    "metrics": {key: fsum(m[key] for m in scoped) / len(scoped) for key in METRICS}})
    overall = {method: {key: fsum(row["metrics"][key] for row in per_repeat if row["method"] == method) / repeats
                        for key in METRICS} for method in methods}
    result = {"repetitions": repeats, "methods": methods, "groups_per_repeat": len(expected_groups),
              "aggregation": "per_auction_totals_then_group_mean_per_repeat_then_arithmetic_repeat_mean",
              "pool_policy": manifest["pool_policy"], "not_implemented": [],
              "per_repeat": per_repeat, "overall": overall,
              "overall_scope": "all_selected_groups_diagnostic_only_use_per_dataset_for_dataset_comparisons",
              "per_dataset": {
                  dataset: {"per_repeat": scoped,
                            "overall": {method: {
                                key: fsum(row["metrics"][key] for row in scoped if row["method"] == method) / repeats
                                for key in METRICS} for method in methods}}
                  for dataset, scoped in dataset_repeats.items()}}
    result["length_realization"] = aggregate_lengths(path, methods=methods)
    atomic_json(path.parent / "summary.json", result)
    return result


def aggregate_lengths(path, *, methods=tuple(METHODS)):
    from utils.length_metrics import summarize_records
    path = Path(path).resolve()
    manifest, methods = load_manifest(path), checked_methods(methods)
    entries, missing, hashes = [], [], {}
    for instance in manifest["instances"]:
        cfg = instance["config"]
        context = {
            "dataset": cfg["dataset"], "repeat_index": instance["repeat_index"],
            "generation_config": {key: cfg[key] for key in ("q", "max_len", "is_structured")}}
        for method in methods:
            selected = selected_path(path.parent, instance, method)
            files = [("selected_reply", None, selected)] + [
                ("per_model_best_sample", name, selected.with_name(
                    f'{name}_results_{instance["instance_id"]}.json')) for name in cfg["models"]]
            for scope, name, file in files:
                source = file.relative_to(path.parent).as_posix()
                if not file.is_file():
                    missing.append({"source": source, "scope": scope})
                    continue
                raw = file.read_bytes()
                record = json.loads(raw)
                assert_paired(record, instance)
                if scope == "selected_reply":
                    name = record["selected_model"]
                hashes[source] = hashlib.sha256(raw).hexdigest()
                entries.append(dict(context, record=record, scope=scope, method=method,
                                    model=name, source=source))
    result = summarize_records(entries)
    result.update(log_files=hashes, missing_logs=missing)
    atomic_json(path.parent / "length_summary.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="write five sets of paired inputs; no LLM calls")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--datasets", nargs="+", choices=["ADGEN", "ATVI"], default=["ADGEN", "ATVI"])
    prepare.add_argument("--groups", nargs="+", help="e.g. ADGEN/jacket_section1 ATVI/Food_section1")
    prepare.add_argument("--categories", nargs="+", help="e.g. ADGEN/jacket ATVI/Food")
    prepare.add_argument("--pool-policy", choices=["stratified_category", "existing_groups"])
    prepare.add_argument("--n-advertisers", type=int, default=20)
    prepare.add_argument("--models", nargs="+")
    prepare.add_argument("--profile", choices=["legacy", "paper"], default="legacy",
                         help="paper: six candidate models and enumerative format; legacy: existing configs")
    prepare.add_argument("--seed", type=int, default=20261027)
    for command in ("run", "aggregate", "lengths"):
        item = sub.add_parser(command)
        item.add_argument("--manifest", required=True)
        item.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    args = parser.parse_args(argv)
    if args.command == "prepare":
        path = prepare_manifest(args.output, datasets=args.datasets, groups=args.groups, models=args.models, seed=args.seed,
                                categories=args.categories, pool_policy=args.pool_policy, n_advertisers=args.n_advertisers,
                                profile=args.profile)
        print(path)
    else:
        fn = {"run": run_manifest, "aggregate": aggregate, "lengths": aggregate_lengths}[args.command]
        print(json.dumps(fn(args.manifest, methods=args.methods), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
