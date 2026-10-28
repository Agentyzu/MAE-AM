import numpy as np


RELEVANCE_FLOOR = 1e-6


def retrieve(model):
    raw = np.asarray(model.scorer.similarities((model.query, text) for text in model.content), dtype=float)
    if raw.shape != (len(model.content),) or not np.all(np.isfinite(raw)):
        raise ValueError("retriever must return one finite cosine per advertiser")
    scores = np.clip(raw, RELEVANCE_FLOOR, 1.0)
    return {
        "query": model.query, "pool_size": len(model.content),
        "raw_cosines": raw.tolist(), "relevance": scores.tolist(),
        "positive_mapping": "clip_cosine_to_[1e-6,1]", "relevance_floor": RELEVANCE_FLOOR,
        "query_scope": "user_query", "document_scope": "complete_original_ad_creative",
        "scorer": model.scorer.metadata(),
        "preselection": "all_supplied_advertisers_remain_eligible",
        "calibration": "semantic_proxy_not_empirically_CTR_calibrated"
    }
