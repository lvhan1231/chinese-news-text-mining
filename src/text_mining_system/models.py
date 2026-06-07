"""Classification models and reporting helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from tqdm import tqdm

from .preprocess import clean_text, tokenize_zh


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float]
    report: pd.DataFrame
    confusion: pd.DataFrame
    predictions: pd.DataFrame


class BertEmbeddingExtractor(BaseEstimator, TransformerMixin):
    """Extract sentence embeddings from a pre-trained Chinese BERT model.

    Uses mean pooling over the last hidden states (excluding padding tokens).
    The model is loaded once and reused across fit/transform calls.
    """

    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        batch_size: int = 32,
        max_length: int = 128,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device

    def _load_model(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self.device is None:
            self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_ = self.device
        self.tokenizer_ = AutoTokenizer.from_pretrained(self.model_name)
        self.model_ = AutoModel.from_pretrained(self.model_name).to(self.device_)
        self.model_.eval()

    def fit(self, X, y=None):
        self._load_model()
        return self

    def transform(self, X):
        import torch

        texts = [str(x) for x in X]
        all_embeddings: list[np.ndarray] = []

        for i in tqdm(range(0, len(texts), self.batch_size), desc="Extracting BERT embeddings"):
            batch_texts = texts[i : i + self.batch_size]
            encoded = self.tokenizer_(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device_)

            with torch.no_grad():
                outputs = self.model_(**encoded)
                # Mean pooling: average of last hidden states, excluding padding
                attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
                embeddings = (outputs.last_hidden_state * attention_mask).sum(dim=1)
                embeddings = embeddings / attention_mask.sum(dim=1).clamp(min=1e-9)
                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)


def build_classifier(kind: str = "hybrid_svm") -> Pipeline:
    """Build a text classifier.

    hybrid_svm combines word-level and character-level TF-IDF features. This is
    robust for Chinese headlines because it keeps segmented words while retaining
    short lexical patterns that may be missed by segmentation.
    """

    word_tfidf = TfidfVectorizer(
        tokenizer=tokenize_zh,
        token_pattern=None,
        ngram_range=(1, 2),
        max_features=60_000,
        sublinear_tf=True,
        min_df=1,
    )

    if kind == "word_lr":
        return Pipeline(
            [
                ("features", word_tfidf),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        n_jobs=None,
                        random_state=42,
                    ),
                ),
            ]
        )

    if kind == "hybrid_svm":
        char_tfidf = TfidfVectorizer(
            analyzer="char",
            preprocessor=clean_text,
            ngram_range=(2, 4),
            max_features=80_000,
            sublinear_tf=True,
            min_df=1,
        )
        features = FeatureUnion(
            [
                ("word", word_tfidf),
                ("char", char_tfidf),
            ]
        )
        return Pipeline(
            [
                ("features", features),
                ("clf", LinearSVC(C=0.5, class_weight="balanced", random_state=42)),
            ]
        )

    if kind == "bert":
        return Pipeline(
            [
                ("features", BertEmbeddingExtractor(batch_size=32, max_length=128)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unsupported classifier kind: {kind}")


def evaluate_classifier(
    model: Pipeline,
    train_texts: pd.Series,
    train_labels: pd.Series,
    eval_texts: pd.Series,
    eval_labels: pd.Series,
) -> EvaluationResult:
    """Fit a classifier and compute standard classification metrics."""

    model.fit(train_texts, train_labels)
    pred = model.predict(eval_texts)
    labels = sorted(set(train_labels.astype(str)) | set(eval_labels.astype(str)))
    metrics = {
        "accuracy": float(accuracy_score(eval_labels, pred)),
        "macro_f1": float(f1_score(eval_labels, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(eval_labels, pred, average="weighted", zero_division=0)),
    }
    report = pd.DataFrame(
        classification_report(
            eval_labels,
            pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
    ).T
    confusion = pd.DataFrame(
        confusion_matrix(eval_labels, pred, labels=labels),
        index=labels,
        columns=labels,
    )
    predictions = pd.DataFrame(
        {
            "text": eval_texts.reset_index(drop=True),
            "gold_label": eval_labels.reset_index(drop=True),
            "pred_label": pd.Series(pred),
        }
    )
    return EvaluationResult(metrics=metrics, report=report, confusion=confusion, predictions=predictions)


def predict_with_scores(model: Pipeline, texts: list[str]) -> pd.DataFrame:
    """Predict labels and expose decision scores when the classifier supports it."""

    labels = model.predict(texts)
    frame = pd.DataFrame({"text": texts, "pred_label": labels})
    clf: BaseEstimator = model.named_steps["clf"]
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(model.named_steps["features"].transform(texts))
        classes = list(getattr(clf, "classes_", []))
        if np.ndim(scores) == 1 and len(classes) == 2:
            scores = np.vstack([-scores, scores]).T
        for idx, class_name in enumerate(classes):
            frame[f"score_{class_name}"] = scores[:, idx]
    return frame


def top_features_by_class(model: Pipeline, top_n: int = 15) -> pd.DataFrame:
    """Return the most positive features for each class in a linear classifier.

    For TF-IDF based models, these are human-readable token/ngram features.
    For BERT embedding models, features are uninterpretable embedding dimensions
    and only dimension indices are reported.
    """

    clf = model.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return pd.DataFrame()

    try:
        features = model.named_steps["features"].get_feature_names_out()
    except Exception:
        features = None

    coefs = clf.coef_
    classes = list(clf.classes_)
    rows: list[dict[str, object]] = []

    if coefs.ndim == 1:
        coefs = coefs.reshape(1, -1)
    if len(classes) == 2 and coefs.shape[0] == 1:
        coefs = np.vstack([-coefs[0], coefs[0]])

    for class_idx, class_name in enumerate(classes):
        order = np.argsort(coefs[class_idx])[-top_n:][::-1]
        for rank, feature_idx in enumerate(order, start=1):
            if features is not None:
                feature_name = str(features[feature_idx]).replace("word__", "").replace("char__", "")
            else:
                feature_name = f"dim_{feature_idx}"
            rows.append(
                {
                    "label": class_name,
                    "rank": rank,
                    "feature": feature_name,
                    "weight": float(coefs[class_idx, feature_idx]),
                }
            )
    return pd.DataFrame(rows)
