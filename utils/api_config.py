import math
import os


def require_env(*names):
    values = [os.getenv(name, "") for name in names]
    missing = [name for name, value in zip(names, values) if not value.strip()]
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")
    return values


def request_timeout():
    timeout = float(os.getenv("API_TIMEOUT_SECONDS", "60"))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("API_TIMEOUT_SECONDS must be finite and positive")
    return timeout
