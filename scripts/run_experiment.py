"""Run reproducible experiments for the Chinese text mining project."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from text_mining_system.data import load_project_dataset
from text_mining_system.models import (
    build_classifier,
    evaluate_classifier,
    top_features_by_class,
)
from text_mining_system.topics import extract_keywords, mine_topics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["sample", "csv", "clue-tnews"], default="sample")
    parser.add_argument("--csv-path", type=Path, help="CSV file with text/label columns.")
    parser.add_argument("--sample-path", type=Path, default=ROOT / "data" / "sample_tnews.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "sample")
    parser.add_argument("--max-train", type=int, default=12000)
    parser.add_argument("--max-eval", type=int, default=3000)
    parser.add_argument("--n-topics", type=int, default=8)
    parser.add_argument("--classifier", choices=["hybrid_svm", "word_lr", "bert"], default="hybrid_svm")
    return parser.parse_args()


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "split" in df.columns and {"train", "validation"}.issubset(set(df["split"])):
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        eval_df = df[df["split"] == "validation"].reset_index(drop=True)
        return train_df, eval_df

    stratify = df["label"] if df["label"].nunique() > 1 else None
    train_df, eval_df = train_test_split(
        df,
        test_size=0.35,
        random_state=42,
        stratify=stratify,
    )
    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)


def save_confusion_plot(confusion: pd.DataFrame, path: Path) -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            break

    plt.figure(figsize=(max(7, len(confusion.columns) * 0.8), max(5, len(confusion.index) * 0.7)))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted label")
    plt.ylabel("Gold label")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()

    # BERT embedding extraction is expensive; use smaller defaults when not overridden
    if args.classifier == "bert":
        default_train = 10000
        default_eval = 2000
        if args.max_train == 12000:  # default wasn't changed
            args.max_train = default_train
        if args.max_eval == 3000:    # default wasn't changed
            args.max_eval = default_eval

    args.output.mkdir(parents=True, exist_ok=True)

    df = load_project_dataset(
        dataset=args.dataset,
        sample_path=args.sample_path,
        csv_path=args.csv_path,
        max_train=args.max_train,
        max_eval=args.max_eval,
    )
    train_df, eval_df = split_dataset(df)

    model = build_classifier(args.classifier)
    result = evaluate_classifier(
        model=model,
        train_texts=train_df["text"],
        train_labels=train_df["label"],
        eval_texts=eval_df["text"],
        eval_labels=eval_df["label"],
    )

    metrics = {
        **result.metrics,
        "dataset": args.dataset,
        "classifier": args.classifier,
        "train_size": int(len(train_df)),
        "eval_size": int(len(eval_df)),
        "num_labels": int(train_df["label"].nunique()),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result.report.to_csv(args.output / "classification_report.csv", encoding="utf-8-sig")
    result.confusion.to_csv(args.output / "confusion_matrix.csv", encoding="utf-8-sig")
    result.predictions.to_csv(args.output / "predictions.csv", index=False, encoding="utf-8-sig")
    save_confusion_plot(result.confusion, args.output / "confusion_matrix.png")

    topics = mine_topics(train_df["text"], n_topics=args.n_topics)
    topics.to_csv(args.output / "topics.csv", index=False, encoding="utf-8-sig")
    keywords = extract_keywords(train_df["text"])
    keywords.to_csv(args.output / "keywords.csv", index=False, encoding="utf-8-sig")
    top_features = top_features_by_class(model)
    top_features.to_csv(args.output / "top_features.csv", index=False, encoding="utf-8-sig")

    joblib.dump(model, args.output / "model.joblib")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved experiment artifacts to: {args.output}")


if __name__ == "__main__":
    main()
