import torch
from typing import Any, List, Tuple
import pandas as pd
import re
from typing import Tuple, List, Any

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

    # Read CSV
    df = pd.read_csv(path)

    # Get URLs
    urls = df['url']

    # Process each URL to extract title and infer label
    X = []
    y = []

    # Clean the URL
    for raw_url in urls:
        clean_url = raw_url.split('?')[0].split('#')[0]

        # Split into segments
        segments = clean_url.split('/')

        # Get the last segment
        last_segment = segments[-1]

        #  If last segment contains 'rcrd' and we have at least 2 segments, use second-to-last
        if 'rcrd' in last_segment and len(segments) >= 2:
            last_segment = segments[-2]

        last_segment = last_segment.replace('.print', '')

        # Remove NBC article IDs
        last_segment = re.sub(r'-(n|rcna|ncna)\d+(-update)?$', '', last_segment)

        # Remove any trailing "rcrd" followed by digits (like rcrd35032)
        last_segment = re.sub(r'rcrd\d+$', '', last_segment)

        # Replace hyphens with spaces
        pseudo_title = last_segment.replace('-', ' ')

        # Remove special characters
        pseudo_title = re.sub(r'[^a-zA-Z0-9\s]', '', pseudo_title)

        # Convert to lowercase
        pseudo_title = pseudo_title.lower()

        # Remove extra spaces
        pseudo_title = ' '.join(pseudo_title.split())

        # Infer label
        label = 'FoxNews' if 'foxnews.com' in raw_url else 'NBC'

        X.append(pseudo_title)
        y.append(label)

    return X, y