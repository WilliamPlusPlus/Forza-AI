from __future__ import annotations

import re
from pathlib import Path


DEFAULT_NAME = "horizon-open-road"
DEFAULT_MODEL_TYPE = "driving"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    slug = slug.strip("-")
    if not slug:
        raise ValueError("Name must contain at least one letter or number.")
    return slug


def data_path(name: str = DEFAULT_NAME, model_type: str = DEFAULT_MODEL_TYPE) -> Path:
    return Path("data") / slugify(model_type) / f"{slugify(name)}.jsonl"


def model_path(name: str = DEFAULT_NAME, model_type: str = DEFAULT_MODEL_TYPE) -> Path:
    return Path("models") / slugify(model_type) / f"{slugify(name)}.joblib"


def online_model_path(name: str = DEFAULT_NAME, model_type: str = DEFAULT_MODEL_TYPE) -> Path:
    return Path("models") / slugify(model_type) / f"{slugify(name)}-online.joblib"
