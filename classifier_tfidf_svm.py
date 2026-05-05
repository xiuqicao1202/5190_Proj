"""
TF-IDF + SVM (Report 2.3.1)
ŷ = sign(w^T v + b), hinge loss; here we use sklearn LinearSVC.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from param_storage import copy_script_source, finetune_base, finetune_prefix, project_root, write_training_txt


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
        "CSV must contain (headline, url) or (processed_title, label). Current columns: "
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
    p.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        help="Project directory containing Finetune_Params/. Defaults to the script's directory.",
    )
    args = p.parse_args()

    proj = project_root(args.project_dir, script_dir=Path(__file__).parent)
    print(f"[paths] project={proj}")

    X, y = load_xy(args.csv)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=args.seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2 / 3, random_state=args.seed, stratify=y_temp)

    model = build_model(args.max_features, args.C)
    model.fit(X_train, y_train)

    metrics: dict = {}
    for name, Xt, yt in [("val", X_val, y_val), ("test", X_test, y_test)]:
        pred = model.predict(Xt)
        acc = accuracy_score(yt, pred)
        f1 = f1_score(yt, pred, pos_label="FoxNews", average="binary")
        metrics[name] = {"accuracy": acc, "f1_fox": f1}
        print(f"{name}: accuracy={acc:.4f} f1_fox={f1:.4f}")

    prefix = finetune_prefix(Path(__file__).stem)
    run_dir = finetune_base(proj) / prefix
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_path = run_dir / f"{prefix}_pipeline.joblib"
    joblib.dump(model, weights_path)
    meta_py = Path(__file__).resolve()
    write_training_txt(
        run_dir / f"{prefix}.txt",
        {
            "Timestamp and description": f"prefix={prefix}",
            "Command line": " ".join(sys.argv),
            "Training hyperparameters": {
                "csv": str(args.csv.resolve()),
                "max_features": args.max_features,
                "C": args.C,
                "seed": args.seed,
            },
            "Validation and test metrics": metrics,
            "Saved files": {"pipeline_joblib": str(weights_path)},
        },
    )
    copy_script_source(meta_py, run_dir / f"{prefix}.py")
    print(f"[save] Finetune_Params/{prefix}/ Weights, {prefix}.txt, and {prefix}.py have been saved.")


if __name__ == "__main__":
    main()
