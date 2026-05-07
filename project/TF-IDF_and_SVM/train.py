import argparse
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from model import NewsClassifier
from preprocess import prepare_data


BASE_DIR = Path(__file__).resolve().parent


def build_matrix(model, texts):
    return torch.stack([model._tfidf(text) for text in texts]).numpy()


def fit_idf(model, texts):
    document_frequency = torch.zeros(model.num_features)

    for text in texts:
        tokens = model._tokens(text)
        features = []

        for token in tokens:
            features.append(model._hash("word:" + token))

        for i in range(len(tokens) - 1):
            features.append(model._hash("bigram:" + tokens[i] + "_" + tokens[i + 1]))

        for index in set(features):
            document_frequency[index] += 1

    n_docs = len(texts)
    model.idf = torch.log((1 + n_docs) / (1 + document_frequency)) + 1


def train_model(X_train, y_train):
    model = NewsClassifier(load_weights=False)
    fit_idf(model, X_train)

    X_matrix = build_matrix(model, X_train)
    svm = LinearSVC(
        class_weight="balanced",
        random_state=42,
        max_iter=10000,
    )
    svm.fit(X_matrix, y_train)

    model.classes = [str(label) for label in svm.classes_]

    if len(model.classes) == 2:
        coef = torch.tensor(svm.coef_, dtype=torch.float32)
        intercept = torch.tensor(svm.intercept_, dtype=torch.float32)
        model.coef = torch.cat([-coef, coef], dim=0)
        model.intercept = torch.cat([-intercept, intercept])
    else:
        model.coef = torch.tensor(svm.coef_, dtype=torch.float32)
        model.intercept = torch.tensor(svm.intercept_, dtype=torch.float32)

    model.eval()
    return model


parser = argparse.ArgumentParser()
parser.add_argument("--data", default=BASE_DIR / "combined_urls.csv")
parser.add_argument("--final", action="store_true")
args = parser.parse_args()

X, y = prepare_data(args.data)

if args.final:
    model = train_model(X, y)
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    model = train_model(X_train, y_train)

torch.save(
    {
        "num_features": model.num_features,
        "classes": model.classes,
        "idf": model.idf,
        "coef": model.coef,
        "intercept": model.intercept,
    },
    BASE_DIR / "model.pt",
)

print("Saved model.pt")
