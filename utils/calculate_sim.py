from collections import OrderedDict
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
import os

import numpy as np


BGE_MODEL = "BAAI/bge-m3"
BGE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


class BGEScorer:
    def __init__(self):
        self._model = None
        self._cache = OrderedDict()

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Install requirements-scoring.txt to run real BGE scoring"
                ) from exc
            self._model = SentenceTransformer(
                BGE_MODEL, revision=BGE_REVISION, device="cpu",
                cache_folder=os.getenv("MAE_AM_BGE_CACHE"),
                local_files_only=os.getenv("MAE_AM_BGE_OFFLINE") == "1",
                trust_remote_code=False, token=False
            )
        return self._model

    def similarities(self, pairs):
        pairs = list(pairs)
        if any(not isinstance(text, str) for pair in pairs for text in pair):
            raise TypeError("BGE inputs must be strings")
        texts = list(dict.fromkeys(text for a, b in pairs if a.strip() and b.strip()
                                   for text in (a, b)))
        vectors = {text: self._cache[text] for text in texts if text in self._cache}
        missing = [text for text in texts if text not in vectors]
        if missing:
            model = self._load_model()
            token_ids = model.tokenizer(missing, truncation=False, padding=False,
                                        add_special_tokens=True,
                                        return_attention_mask=False)["input_ids"]
            if any(len(ids) > model.max_seq_length for ids in token_ids):
                raise ValueError(f"BGE input exceeds {model.max_seq_length} tokens; text was not truncated")
            embeddings = np.asarray(model.encode(
                missing, batch_size=8, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False
            ), dtype=np.float64)
            if embeddings.ndim != 2 or len(embeddings) != len(missing):
                raise ValueError("BGE returned an unexpected embedding shape")
            norms = np.linalg.norm(embeddings, axis=1)
            if not np.all(np.isfinite(embeddings)) or np.any(norms <= 0):
                raise ValueError("BGE returned non-finite or zero embeddings")
            for text, embedding in zip(missing, embeddings / norms[:, None]):
                vectors[text] = embedding
                self._cache[text] = embedding
            while len(self._cache) > 1024:
                self._cache.popitem(last=False)
        return [float(np.clip(np.dot(vectors[a], vectors[b]), -1., 1.))
                if a.strip() and b.strip() else 0.0 for a, b in pairs]

    def metadata(self):
        libraries = {}
        for name in ("sentence-transformers", "transformers", "torch"):
            try:
                libraries[name] = version(name)
            except PackageNotFoundError:
                libraries[name] = None
        resolved = None
        if self._model is not None:
            config = getattr(getattr(self._model[0], "auto_model", None), "config", None)
            resolved = getattr(config, "_commit_hash", None)
        return {"model": BGE_MODEL, "requested_revision": BGE_REVISION,
                "resolved_revision": resolved, "metric": "dense_embedding_cosine",
                "device": "cpu", "empty_text_score": 0.0,
                "overlength_policy": "reject_without_truncation", "libraries": libraries}


@lru_cache(maxsize=1)
def get_scorer():
    return BGEScorer()


def sim(text1, text2):
    return get_scorer().similarities([(text1, text2)])[0]
