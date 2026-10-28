from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import os
import re
from urllib.parse import urlsplit, urlunsplit


SPECS = {
    "ChatGPT": ("GPT-5", "gpt-5-2025-08-07", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "claude": ("Claude 4", None, None, "CLAUDE_API_KEY"),
    "kimi": ("Kimi K2.6", "kimi-k2.6", "https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"),
    "GLM": ("GLM-5.1", "glm-5.1", "https://open.bigmodel.cn/api/paas/v4", "ZHIPUAI_API_KEY"),
    "ernie": ("ERNIE 5.0", "ernie-5.0", "https://qianfan.baidubce.com/v2", "QIANFAN_API_KEY"),
    "Qwen": ("Qwen3", None, "https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
}
PATTERNS = {
    "ChatGPT": r"gpt-5(?:-\d{4}-\d{2}-\d{2})?",
    "claude": r"claude-(?:sonnet|opus)-4-20250514",
    "kimi": r"kimi-k2\.6", "GLM": r"glm-5\.1", "ernie": r"ernie-5\.0",
    "Qwen": r"qwen3-(?:235b-a22b|30b-a3b|32b|14b|8b|4b|1\.7b|0\.6b)(?:-(?:instruct|thinking)-2507)?",
}


def settings(model, name):
    family, model_id, base_url, key_env = SPECS[name]
    entry = deepcopy(getattr(model, "backbones", {}).get(name, {}))
    unknown = set(entry) - {"model_id", "base_url", "parameters"}
    if unknown:
        raise ValueError(f"Unsupported backbone settings for {name}: {sorted(unknown)}")
    prefix = "MAE_AM_" + name.upper()
    model_id = entry.get("model_id", os.getenv(prefix + "_MODEL_ID", model_id))
    base_url = entry.get("base_url", os.getenv(prefix + "_BASE_URL", base_url))
    if not model_id:
        raise ValueError(f"Set backbones.{name}.model_id or {prefix}_MODEL_ID to an explicit {family} variant")
    if not re.fullmatch(PATTERNS[name], model_id):
        raise ValueError(f"{name} model_id does not identify the paper's {family}: {model_id}")
    if not base_url:
        raise ValueError(f"Set backbones.{name}.base_url or {prefix}_BASE_URL to a service providing this model")
    url = urlsplit(base_url)
    if url.scheme not in ("https", "http") or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("base_url must be an HTTP(S) API endpoint without credentials or query parameters")
    base_url = urlunsplit((url.scheme, url.netloc, url.path.rstrip("/"), "", ""))
    parameters = entry.get("parameters", {})
    allowed = {"temperature", "top_p", "max_tokens", "max_completion_tokens", "reasoning_effort", "extra_body"}
    if name == "GLM":
        allowed -= {"max_completion_tokens", "reasoning_effort"}
    if not isinstance(parameters, dict) or set(parameters) - allowed:
        raise ValueError(f"Unsupported generation parameters for {name}")
    extra = parameters.get("extra_body", {})
    if not isinstance(extra, dict) or set(extra) - {"thinking", "enable_thinking", "thinking_budget"}:
        raise ValueError("extra_body may configure thinking only")
    if name == "kimi":
        thinking = extra.get("thinking", {"type": "enabled"})
        if thinking not in ({"type": "enabled"}, {"type": "disabled"}):
            raise ValueError("Kimi thinking.type must be enabled or disabled")
        expected = .6 if thinking["type"] == "disabled" else 1.
        if parameters.get("temperature", expected) != expected or parameters.get("top_p", .95) != .95:
            raise ValueError("Kimi K2.6 requires temperature=1 (thinking) / 0.6 (nonthinking), top_p=0.95")
    return dict(backend=name, paper_backbone=family, requested_model=model_id,
                base_url=base_url, parameters=parameters), key_env


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def reply(model, query, name):
    spec, key_env = settings(model, name)
    api_key = os.getenv(key_env)
    if not api_key:
        raise ValueError(f"Set {key_env} before calling {name}")
    request = dict(model=spec["requested_model"], messages=deepcopy(model.messages) +
                   [{"role": "user", "content": query}], **spec["parameters"])
    sdk = "zhipuai" if name == "GLM" else "openai"
    if name == "GLM":
        from zhipuai import ZhipuAI
        client = ZhipuAI(api_key=api_key, base_url=spec["base_url"])
    else:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=spec["base_url"])
    called_at = datetime.now(timezone.utc).isoformat()
    response = client.chat.completions.create(**request)
    choice = _get(response, "choices")[0]
    text = _get(_get(choice, "message"), "content")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{name} returned no text answer")
    try:
        sdk_version = version(sdk)
    except PackageNotFoundError:
        sdk_version = None
    record = dict(spec, called_at_utc=called_at,
                  returned_model=_get(response, "model"), response_id=_get(response, "id"),
                  system_fingerprint=_get(response, "system_fingerprint"),
                  finish_reason=_get(choice, "finish_reason"), sdk=sdk, sdk_version=sdk_version)
    model.llm_calls.append(record)
    model.messages.extend([{"role": "user", "content": query}, {"role": "assistant", "content": text}])
    return text

