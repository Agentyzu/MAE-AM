from copy import deepcopy
from datetime import datetime, timezone
from math import fsum
import os
from urllib.parse import urlsplit

import numpy as np

from auction.LERA import shortlist, single_round


def settings(config, q):
    if not isinstance(config, dict):
        raise ValueError("LERA configuration must be a mapping")
    result = dict(base_url=os.getenv("MAE_AM_LERA_BASE_URL"), model_id=os.getenv("MAE_AM_LERA_MODEL_ID"),
                  top_k=5 if q == 1 else 8, max_segments=10, timeout=60.,
                  chat_template_kwargs={"enable_thinking": False})
    if set(config) - set(result):
        raise ValueError("Unsupported LERA setting")
    result.update(deepcopy(config))
    if not result["base_url"] or not result["model_id"]:
        raise ValueError("Set lera.base_url/model_id or MAE_AM_LERA_BASE_URL/MAE_AM_LERA_MODEL_ID")
    url = urlsplit(result["base_url"])
    if (url.scheme not in ("http", "https") or not url.hostname or url.username or url.password
            or url.query or url.fragment):
        raise ValueError("LERA URL must be an HTTP(S) endpoint without embedded credentials")
    result["base_url"] = result["base_url"].rstrip("/")
    for key in ("top_k", "max_segments"):
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < 1:
            raise ValueError(f"LERA {key} must be a positive integer")
    if not np.isfinite(result["timeout"]) or not 0 < result["timeout"] <= 60:
        raise ValueError("LERA timeout must lie in (0,60]")
    if not isinstance(result["chat_template_kwargs"], dict):
        raise ValueError("LERA chat template kwargs must be a mapping")
    return result


class SGLangScorer:
    def __init__(self, config):
        self.config = config
        self.info = None
        self.calls = []
        self.choice_tokens = {}

    def _request(self, path, payload=None):
        import httpx
        key = os.getenv("MAE_AM_LERA_API_KEY")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        with httpx.Client(timeout=self.config["timeout"], headers=headers) as client:
            response = client.request("GET" if payload is None else "POST",
                                      self.config["base_url"] + path, json=payload)
            response.raise_for_status()
            return response.json()

    def metadata(self):
        if self.info is None:
            self.info = self._request("/get_model_info")
            names = [self.info.get(k) for k in ("model_path", "served_model_name")]
            if self.config["model_id"] not in names:
                raise ValueError("SGLang reported model differs from configured LERA model_id")
        return dict(settings=deepcopy(self.config), model_info=deepcopy(self.info))

    def _tokens(self, text=None, *, prompt=None):
        if prompt is None and text in self.choice_tokens:
            return list(self.choice_tokens[text])
        payload = dict(model=self.config["model_id"], add_special_tokens=False)
        if prompt is None:
            payload["prompt"] = text
        else:
            payload.update(messages=[{"role": "user", "content": prompt}], add_generation_prompt=True,
                           chat_template_kwargs=self.config["chat_template_kwargs"])
        ids = self._request("/tokenize", payload).get("tokens")
        if (not isinstance(ids, list) or not ids or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids)):
            raise ValueError("SGLang returned invalid token IDs")
        if prompt is None:
            self.choice_tokens[text] = list(ids)
        return ids

    def generate(self, prompt, *, purpose):
        self.metadata()
        ids = self._tokens(prompt=prompt)
        result = self._request("/generate", dict(input_ids=ids,
                               sampling_params={"max_new_tokens": 64, "temperature": .7}))
        text = result.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("SGLang returned no LERA keyword/segment text")
        self.calls.append(dict(purpose=purpose, prompt=prompt, text=text,
                               called_at_utc=datetime.now(timezone.utc).isoformat(), meta_info=result.get("meta_info")))
        return text.strip()

    def score(self, prompt, labels):
        self.metadata()
        prefix = self._tokens(prompt=prompt)
        suffixes = [self._tokens(label) for label in labels]
        payload = dict(input_ids=[prefix + suffix for suffix in suffixes],
                       sampling_params={"max_new_tokens": 0, "temperature": 0.},
                       return_logprob=True, logprob_start_len=max(len(prefix)-1, 0))
        responses = self._request("/generate", payload)
        if not isinstance(responses, list) or len(responses) != len(labels):
            raise ValueError("SGLang must return one prefill result for every candidate and no-insertion label")
        logits, records = [], []
        for response, ids in zip(responses, suffixes):
            logs = response.get("meta_info", {}).get("input_token_logprobs", [])
            tail = logs[-len(ids):]
            if (len(tail) != len(ids) or any(len(row) < 2 or row[1] != token
                    or row[0] is None or not np.isfinite(row[0]) or row[0] > 1e-6
                    for row, token in zip(tail, ids))):
                raise ValueError("SGLang is missing actual candidate-token logprobs or has a token-boundary mismatch")
            logits.append(fsum(float(row[0]) for row in tail))
            records.append(dict(choice_token_ids=ids, choice_token_logprobs=tail,
                                response_id=response.get("meta_info", {}).get("id")))
        self.calls.append(dict(purpose="choice_sequence_logprobs", prompt=prompt, labels=labels,
                               prefix_token_count=len(prefix), choice_logprobs=logits, choices=records,
                               called_at_utc=datetime.now(timezone.utc).isoformat()))
        return logits


def prepare(model, client):
    cfg = client.config
    rounds, inserted, prefix = [], [], ""
    client.calls = []
    language = "Chinese" if model.language == "zh" else "English"
    for t in range(cfg["max_segments"]):
        if len(inserted) >= model.q:
            break
        eligible = [i for i in range(len(model.b)) if i not in inserted]
        if not eligible:
            break
        keyword_prompt = (f"Extract two or three product-category keywords for the next two or three sentences. "
                          f"Account for user needs and avoid products already addressed. Output only comma-separated "
                          f"keywords in {language}.\nUser query: {model.query}\nCurrent response prefix: {prefix}")
        keywords = client.generate(keyword_prompt, purpose="keywords")
        raw = np.asarray(model.scorer.similarities((keywords, model.content[i]) for i in eligible), float)
        if raw.shape != (len(eligible),) or not np.all(np.isfinite(raw)) or np.any(abs(raw) > 1+1e-8):
            raise ValueError("LERA retriever must return one finite cosine per eligible advertiser")
        relevance = np.clip((raw + 1) / 2, 1e-6, 1.)
        k = min(cfg["top_k"], len(eligible))
        candidates, _ = shortlist(np.asarray(model.b)[eligible], relevance, k)
        prompt = ("Decide which advertisement best fits the next few sentences, or select no insertion. "
                  "Follow user preferences and commercial intent. Output only one choice label.\n"
                  f"User query: {model.query}\nCurrent response prefix: {prefix}\nChoice0: No insertion\n")
        for label, index in enumerate(candidates, 1):
            i = eligible[index]
            prompt += f"Choice{label}: {model.ad_name[i]}: {model.content[i]}\n"
        labels = [f"Choice{i}" for i in range(k+1)]
        logprobs = client.score(prompt, labels)
        outcome = single_round(np.asarray(model.b)[eligible], relevance, logprobs, k)
        winner = None if outcome["winner"] is None else eligible[outcome["winner"]]
        continuation = (f"Continue the assistant response in {language} within two or three sentences. "
                        f"Maintain coherence and do not repeat earlier content.\nUser query: {model.query}\n"
                        f"Current response prefix: {prefix}\n")
        if winner is None:
            continuation += "No ad was selected. Do not insert an advertisement or mention any advertiser."
        else:
            continuation += (f"Naturally incorporate {model.ad_name[winner]} and no other brand. "
                             f"Advertiser description: {model.content[winner]}")
            inserted.append(winner)
        segment = client.generate(continuation, purpose="dynamic_prefix_segment")
        rounds.append(dict(segment_index=t, eligible_indices=eligible, k=k, keywords=keywords,
                           raw_cosines=raw.tolist(), relevance=relevance.tolist(),
                           choice_logprobs=logprobs, winner=winner, prefix_before=prefix, segment=segment))
        prefix = (prefix + "\n" + segment).strip()
    return dict(rounds=rounds, service=client.metadata(), model_calls=deepcopy(client.calls),
                scorer=model.scorer.metadata(), normalization="clip((cosine+1)/2,1e-6,1)",
                duplicate_policy="previously_inserted_advertisers_ineligible",
                final_generation="freeze_insertion_order_then_common_five_candidates_per_model")
