import re
from typing import List, Tuple

import pandas as pd


def prepare_data(path: str) -> Tuple[List[str], List[str]]:
    """
    Convert article URLs into pseudo-headline text and source labels.

    This matches the TF-IDF workflow: X is a list of strings and y is a list
    of labels aligned with X.
    """
    df = pd.read_csv(path)
    urls = df["url"]

    X = []
    y = []

    for raw_url in urls:
        clean_url = raw_url.split("?")[0].split("#")[0]
        segments = clean_url.split("/")
        last_segment = segments[-1]

        if "rcrd" in last_segment and len(segments) >= 2:
            last_segment = segments[-2]

        last_segment = last_segment.replace(".print", "")
        last_segment = re.sub(r"-(n|rcna|ncna)\d+(-update)?$", "", last_segment)
        last_segment = re.sub(r"rcrd\d+$", "", last_segment)

        pseudo_title = last_segment.replace("-", " ")
        pseudo_title = re.sub(r"[^a-zA-Z0-9\s]", "", pseudo_title)
        pseudo_title = " ".join(pseudo_title.lower().split())

        label = "FoxNews" if "foxnews.com" in raw_url else "NBC"

        X.append(pseudo_title)
        y.append(label)

    return X, y
