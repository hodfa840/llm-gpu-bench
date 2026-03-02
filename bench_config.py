"""
bench_config.py — shared config loader for all benchmark scripts.
Reads config.env from the same directory as this file.
Sets HF_TOKEN in the environment so HuggingFace libraries pick it up.
"""

import os
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "config.env"


def load_config() -> dict:
    """Parse config.env and return a dict of key→value."""
    cfg = {}
    if not CONFIG_FILE.exists():
        return cfg
    for line in CONFIG_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        cfg[key.strip()] = val.strip()
    return cfg


def apply_hf_token(cfg: dict) -> bool:
    """
    Looks for HF token in this order:
      1. HF_TOKEN key in config.env
      2. hf_token file in the same directory as this script
    Injects it into the environment so HuggingFace libraries pick it up.
    Returns True if a token was applied.
    """
    token = cfg.get("HF_TOKEN", "").strip()

    # Fallback: read from hf_token file next to this script
    if not token:
        token_file = Path(__file__).parent / "hf_token"
        if token_file.exists():
            token = token_file.read_text().strip()

    if token:
        os.environ["HF_TOKEN"]               = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        # Inject directly into huggingface_hub without any network call
        try:
            import huggingface_hub.constants as _hf_const
            _hf_const.HF_TOKEN_PATH  # just import check, no login needed
        except Exception:
            pass
        return True
    return False


# ── Convenience accessor ──────────────────────────────────────────────────────

def get(key: str, default: str = "") -> str:
    return _CACHE.get(key, default)

def get_bool(key: str, default: bool = False) -> bool:
    v = _CACHE.get(key, str(default)).lower()
    return v in ("1", "true", "yes")

def get_int(key: str, default: int = 0) -> int:
    try:
        return int(_CACHE.get(key, str(default)))
    except ValueError:
        return default

def get_list(key: str, default: list = None) -> list:
    val = _CACHE.get(key, "")
    return [x for x in val.split() if x] if val else (default or [])


# ── Auto-load on import ───────────────────────────────────────────────────────

_CACHE = load_config()
_token_applied = apply_hf_token(_CACHE)


def print_config_summary():
    token_set   = bool(_CACHE.get("HF_TOKEN", ""))
    email_to    = _CACHE.get("EMAIL_TO", "(not set)")
    smtp_host   = _CACHE.get("SMTP_HOST", "(local sendmail)")
    families    = get_list("DEFAULT_FAMILIES")
    max_gen     = get_int("MAX_NEW_TOKENS", 200)

    print(f"  Config file : {CONFIG_FILE}")
    print(f"  HF token    : {'✓ loaded' if token_set else '✗ not set  → gated models will be skipped'}")
    print(f"  Email to    : {email_to}")
    print(f"  SMTP host   : {smtp_host or '(local sendmail)'}")
    print(f"  Families    : {', '.join(families) if families else 'ALL'}")
    print(f"  Max tokens  : {max_gen}")
