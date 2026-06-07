"""Streamlit demo for the Chinese text mining project."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from text_mining_system.data import load_sample_csv
from text_mining_system.labels import label_to_display
from text_mining_system.models import build_classifier, predict_with_scores
from text_mining_system.similarity import SimilarityIndex
from text_mining_system.topics import mine_topics


st.set_page_config(page_title="中文新闻文本挖掘系统", page_icon="🧭", layout="wide")


@st.cache_resource
def load_demo_resources():
    df = load_sample_csv(ROOT / "data" / "sample_tnews.csv")
    full_model_path = ROOT / "results" / "tnews_full" / "model.joblib"
    if full_model_path.exists():
        model = joblib.load(full_model_path)
    else:
        model = build_classifier("hybrid_svm")
        model.fit(df["text"], df["label"])
    index = SimilarityIndex().fit(df["text"], df["label"])
    topics = mine_topics(df["text"], n_topics=5)
    return df, model, index, topics


df, model, index, topics = load_demo_resources()

st.title("中文新闻文本挖掘系统")

tab_predict, tab_search, tab_topics, tab_data = st.tabs(["分类预测", "相似检索", "主题发现", "样例数据"])

with tab_predict:
    text = st.text_area(
        "新闻标题",
        value="人工智能公司发布新一代中文文本理解模型",
        height=120,
    )
    if st.button("预测类别", type="primary"):
        pred = predict_with_scores(model, [text])
        pred_label = pred.loc[0, "pred_label"]
        st.subheader(f"预测结果：{label_to_display(pred_label)}")
        score_cols = [col for col in pred.columns if col.startswith("score_")]
        if score_cols:
            scores = (
                pred[score_cols]
                .T.reset_index()
                .rename(columns={"index": "类别", 0: "得分"})
            )
            scores["类别"] = scores["类别"].str.replace("score_", "", regex=False)
            scores["类别"] = scores["类别"].map(label_to_display)
            st.bar_chart(scores.set_index("类别"))

with tab_search:
    query = st.text_input("检索文本", value="芯片和人工智能产业发展")
    top_k = st.slider("返回数量", 3, 10, 5)
    result = index.search(query, top_k=top_k)
    if "label" in result.columns:
        result["label"] = result["label"].map(label_to_display)
    result = result.rename(columns={"score": "相似度", "text": "新闻标题", "label": "类别"})
    st.dataframe(result, width="stretch", hide_index=True)

with tab_topics:
    st.dataframe(topics, width="stretch", hide_index=True)

with tab_data:
    display_labels = df["label"].map(label_to_display)
    label_counts = display_labels.value_counts().rename_axis("类别").reset_index(name="数量")
    display_df = df.copy()
    display_df["label"] = display_df["label"].map(label_to_display)
    display_df = display_df.rename(columns={"text": "新闻标题", "label": "类别"})
    left, right = st.columns([1, 2])
    with left:
        st.bar_chart(label_counts.set_index("类别"))
    with right:
        st.dataframe(display_df, width="stretch", hide_index=True)
