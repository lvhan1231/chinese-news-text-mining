"""Unsupervised topic mining and keyword extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocess import tokenize_zh


def _word_vectorizer(max_features: int = 30_000) -> TfidfVectorizer:
    return TfidfVectorizer(
        tokenizer=tokenize_zh,
        token_pattern=None,
        max_features=max_features,
        sublinear_tf=True,
        min_df=1,
        max_df=0.95,
    )


def mine_topics(texts: pd.Series, n_topics: int = 8, top_words: int = 12) -> pd.DataFrame:
    """Mine latent topics with NMF over word TF-IDF features."""

    clean_texts = texts.astype(str).reset_index(drop=True)
    vectorizer = _word_vectorizer()
    matrix = vectorizer.fit_transform(clean_texts)
    feature_names = np.array(vectorizer.get_feature_names_out())
    topic_count = max(2, min(n_topics, matrix.shape[0] - 1, matrix.shape[1] - 1))

    nmf = NMF(
        n_components=topic_count,
        init="nndsvda",
        random_state=42,
        max_iter=500,
        l1_ratio=0.1,
    )
    nmf.fit(matrix)

    rows: list[dict[str, object]] = []
    for topic_idx, weights in enumerate(nmf.components_):
        order = np.argsort(weights)[-top_words:][::-1]
        rows.append(
            {
                "topic_id": topic_idx,
                "top_words": " / ".join(feature_names[order]),
                "top_weight": float(weights[order[0]]),
            }
        )
    return pd.DataFrame(rows)


def extract_keywords(texts: pd.Series, top_n: int = 8, limit: int = 80) -> pd.DataFrame:
    """Extract representative TF-IDF keywords for sample documents."""

    sample = texts.astype(str).head(limit).reset_index(drop=True)
    vectorizer = _word_vectorizer()
    matrix = vectorizer.fit_transform(sample)
    feature_names = np.array(vectorizer.get_feature_names_out())
    rows: list[dict[str, object]] = []

    for row_idx in range(matrix.shape[0]):
        row = matrix.getrow(row_idx)
        if row.nnz == 0:
            keywords = ""
        else:
            local_order = np.argsort(row.data)[-top_n:][::-1]
            keywords = " / ".join(feature_names[row.indices[local_order]])
        rows.append({"text": sample.iloc[row_idx], "keywords": keywords})
    return pd.DataFrame(rows)
