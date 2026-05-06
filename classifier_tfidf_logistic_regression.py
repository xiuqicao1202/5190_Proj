"""
TF-IDF + Logistic Regression (Report 2.3.1)
ŷ = σ(w^T v + b), binary cross-entropy loss.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from param_storage import copy_script_source, finetune_base, finetune_prefix, project_root, write_training_txt

# 固定文件名，供 params_submit/model.py / 评测载入（复制整个提交目录时请带上 LR_submit/）
LR_SUBMIT_PIPELINE_NAME = "tfidf_lr_pipeline.joblib"


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
        "CSV must contain (headline, url) or (processed_title, label) columns. Current columns: "
        + ", ".join(map(str, df.columns))
    )


def build_model(max_features: int | None, C: float) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english",max_features=max_features)),
            ("clf", LogisticRegression(max_iter=100, C=C, solver="liblinear")),
        ]
    )


def main() -> None:
    # 典型命令行如下：
    # python classifier_tfidf_logistic_regression.py --csv ./Newsheadlines/url_with_headlines.csv --max-features 100 --C 1.0 --seed 42
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path(__file__).parent / "Newsheadlines" / "url_with_headlines.csv")
    p.add_argument("--max-features", type=int, default=100)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        help="Project directory containing Pretrained_Params/ and Finetune_Params/; defaults to script directory.",
    )
    p.add_argument(
        "--lr-submit-dir",
        type=Path,
        default=None,
        help="Directory to save a copy of the trained pipeline for leaderboard submit (LR_submit). "
        f"Defaults to <project-dir>/LR_submit ({LR_SUBMIT_PIPELINE_NAME}). Use --skip-lr-submit to disable.",
    )
    p.add_argument(
        "--skip-lr-submit",
        action="store_true",
        help=f"Do not write {LR_SUBMIT_PIPELINE_NAME} under LR_submit.",
    )
    args = p.parse_args()

    proj = project_root(args.project_dir, script_dir=Path(__file__).parent)
    print(f"[paths] project={proj} (TF-IDF does not use any external pretrained weights, only saved to Finetune_Params/)")

    X, y = load_xy(args.csv)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=args.seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=2 / 3, random_state=args.seed, stratify=y_temp
    )
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
    lr_submit_dir = (
        args.lr_submit_dir.expanduser().resolve()
        if args.lr_submit_dir is not None
        else (proj / "LR_submit").resolve()
    )
    lr_submit_pipeline: Path | None = None
    if not args.skip_lr_submit:
        lr_submit_dir.mkdir(parents=True, exist_ok=True)
        lr_submit_pipeline = lr_submit_dir / LR_SUBMIT_PIPELINE_NAME
        joblib.dump(model, lr_submit_pipeline)
        print(f"[save] LR_submit/: {lr_submit_pipeline}")
        # 与 classifier_roberta.py 一致：再复制一份到 params_submit，供仅提交 params_submit 时 model.py 同目录加载
        submit_dir = proj / "params_submit"
        submit_dir.mkdir(parents=True, exist_ok=True)
        submit_pipeline = submit_dir / LR_SUBMIT_PIPELINE_NAME
        shutil.copy2(lr_submit_pipeline, submit_pipeline)
        print(f"[save] params_submit/ (mirror): {submit_pipeline}")
    meta_py = Path(__file__).resolve()
    saved_section: dict = {"pipeline_joblib": str(weights_path)}
    if lr_submit_pipeline is not None:
        saved_section["lr_submit_pipeline_joblib"] = str(lr_submit_pipeline.resolve())
        saved_section["params_submit_pipeline_joblib"] = str(
            (proj / "params_submit" / LR_SUBMIT_PIPELINE_NAME).resolve()
        )
    write_training_txt(
        run_dir / f"{prefix}.txt",
        {
            "timestamp_and_comment": f"prefix={prefix}",
            "command_line": " ".join(sys.argv),
            "training_hyperparameters": {
                "csv": str(args.csv.resolve()),
                "max_features": args.max_features,
                "C": args.C,
                "seed": args.seed,
            },
            "validation_and_test_metrics": metrics,
            "saved_files": saved_section,
        },
    )
    copy_script_source(meta_py, run_dir / f"{prefix}.py")
    print(f"[save] Finetune_Params/{prefix}/ model weights, {prefix}.txt, and {prefix}.py have been saved.")


if __name__ == "__main__":
    main()
