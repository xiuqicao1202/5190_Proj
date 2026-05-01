"""
TF-IDF + SVM（报告 2.3.1）
ˆy = sign(w^T v + b)，hinge 损失；此处用 sklearn LinearSVC。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def label_from_url(url: str) -> str:
    return "FoxNews" if "foxnews.com" in url.lower() else "NBC"


def load_xy(csv_path: Path) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(csv_path)
    if "processed_title" in df.columns and "label" in df.columns:
        texts = df["processed_title"].fillna("").astype(str).tolist()
        labels = df["label"].astype(str).tolist()
        return texts, labels
    if "headline" in df.columns and "url" in df.columns:
        texts = df["headline"].fillna("").astype(str).tolist()
        labels = [label_from_url(str(u)) for u in df["url"].tolist()]
        return texts, labels
    raise ValueError(
        "CSV 需要包含 (headline, url) 或 (processed_title, label)，当前列: "
        + ", ".join(map(str, df.columns))
    )


def build_model(max_features: int | None, C: float) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))),
            ("clf", LinearSVC(C=C, max_iter=5000)),
        ]
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path(__file__).parent / "Newsheadlines" / "url_with_headlines.csv")
    p.add_argument("--max-features", type=int, default=20000)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    X, y = load_xy(args.csv)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=args.seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2 / 3, random_state=args.seed, stratify=y_temp)

    model = build_model(args.max_features, args.C)
    model.fit(X_train, y_train)

    for name, Xt, yt in [("val", X_val, y_val), ("test", X_test, y_test)]:
        pred = model.predict(Xt)
        acc = accuracy_score(yt, pred)
        f1 = f1_score(yt, pred, pos_label="FoxNews", average="binary")
        print(f"{name}: accuracy={acc:.4f} f1_fox={f1:.4f}")


if __name__ == "__main__":
    main()
