import pandas as pd
from typing import List, Tuple


def prepare_data(path: str) -> Tuple[List[str], List[str]]:
    """
    Template preprocessing for leaderboard.

    Requirements:
    - Must read the provided templates path at `path`.
    - Must return a tuple (X, y):
        X: a list of model-ready inputs (these must match what your model expects in predict(...))
        y: a list of ground-truth labels aligned with X (same length)

    Notes:
    - The evaluation backend will call this function with the shared validation templates
    - Ensure the output format (types, shapes) of X matches your model's predict(...) inputs.
    """

    df = pd.read_csv(path)

    X: List[str] = df["headline"].fillna("").astype(str).tolist()
    y: List[str] = [
        "FoxNews" if "foxnews.com" in str(u).lower() else "NBC"
        for u in df["url"].tolist()
    ]

    return X, y


if __name__ == "__main__":
    X, y = prepare_data("url_with_headlines.csv")
    df_out = pd.DataFrame({"headline": X, "label": y})
    df_out.to_csv("after_preprocess.csv", index=False, encoding="utf-8")
