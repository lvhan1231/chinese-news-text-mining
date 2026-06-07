"""Dataset loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TEXT_CANDIDATES = ("sentence", "text", "content", "title")
LABEL_CANDIDATES = ("label_desc", "label", "category")


def _pick_column(columns: list[str], candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not find one of columns {candidates}; got {columns}")


def _normalize_frame(df: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    columns = list(df.columns)
    text_col = _pick_column(columns, TEXT_CANDIDATES)
    label_col = _pick_column(columns, LABEL_CANDIDATES)
    out = pd.DataFrame(
        {
            "text": df[text_col].astype(str),
            "label": df[label_col].astype(str),
        }
    )
    if split:
        out["split"] = split
    out = out.dropna()
    out = out[out["text"].str.strip().astype(bool)]
    out = out[out["label"].str.strip().astype(bool)]
    return out.reset_index(drop=True)


def load_sample_csv(path: str | Path) -> pd.DataFrame:
    """Load the small repository sample used for smoke tests."""

    return _normalize_frame(pd.read_csv(path))


def _label_to_name(dataset_split, frame: pd.DataFrame, label_col: str) -> pd.Series:
    feature = dataset_split.features.get(label_col)
    if hasattr(feature, "names") and feature.names:
        names = feature.names
        return frame[label_col].map(lambda value: names[int(value)])
    return frame[label_col].astype(str)


def load_clue_tnews(max_train: int | None = None, max_eval: int | None = None) -> pd.DataFrame:
    """Load the TNEWS subset from Hugging Face CLUE datasets."""

    from datasets import load_dataset

    errors: list[str] = []
    dataset = None
    candidates = (
        ("clue/clue", "tnews", {}),
        ("clue", "tnews", {"trust_remote_code": True}),
    )
    for path, name, kwargs in candidates:
        try:
            dataset = load_dataset(path, name, **kwargs)
            break
        except Exception as exc:  # pragma: no cover - depends on network/cache.
            errors.append(f"{path}/{name}: {exc}")

    if dataset is None:
        details = "\n".join(errors)
        raise RuntimeError(f"Unable to load CLUE TNEWS from Hugging Face:\n{details}")

    frames: list[pd.DataFrame] = []
    split_limits = {"train": max_train, "validation": max_eval}
    for split_name in ("train", "validation"):
        if split_name not in dataset:
            continue
        split = dataset[split_name]
        limit = split_limits[split_name]
        if limit:
            split = split.select(range(min(limit, len(split))))
        frame = split.to_pandas()
        text_col = _pick_column(list(frame.columns), TEXT_CANDIDATES)
        label_col = _pick_column(list(frame.columns), LABEL_CANDIDATES)
        if label_col == "label" and "label_desc" not in frame.columns:
            frame["label"] = _label_to_name(split, frame, label_col)
        frames.append(_normalize_frame(frame, split=split_name))

    if not frames:
        raise RuntimeError("The loaded dataset does not contain train or validation splits.")
    return pd.concat(frames, ignore_index=True)


def load_project_dataset(
    dataset: str,
    sample_path: str | Path,
    csv_path: str | Path | None = None,
    max_train: int | None = None,
    max_eval: int | None = None,
) -> pd.DataFrame:
    """Load a dataset by CLI-friendly name."""

    if dataset == "sample":
        return load_sample_csv(sample_path)
    if dataset == "csv":
        if csv_path is None:
            raise ValueError("--csv-path is required when --dataset csv is used.")
        return _normalize_frame(pd.read_csv(csv_path))
    if dataset == "clue-tnews":
        return load_clue_tnews(max_train=max_train, max_eval=max_eval)
    raise ValueError(f"Unsupported dataset: {dataset}")
