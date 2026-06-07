"""Similarity search for Chinese news headlines."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import tokenize_zh


class SimilarityIndex:
    """A lightweight TF-IDF cosine similarity index."""

    def __init__(self, max_features: int = 50_000) -> None:
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize_zh,
            token_pattern=None,
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
        )
        self.matrix = None
        self.items = pd.DataFrame()

    def fit(self, texts: pd.Series, labels: pd.Series | None = None) -> "SimilarityIndex":
        self.items = pd.DataFrame({"text": texts.astype(str).reset_index(drop=True)})
        if labels is not None:
            self.items["label"] = labels.astype(str).reset_index(drop=True)
        self.matrix = self.vectorizer.fit_transform(self.items["text"])
        return self

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        if self.matrix is None:
            raise RuntimeError("SimilarityIndex must be fitted before search.")
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).ravel()
        order = scores.argsort()[-top_k:][::-1]
        result = self.items.iloc[order].copy().reset_index(drop=True)
        result.insert(0, "score", scores[order])
        return result
