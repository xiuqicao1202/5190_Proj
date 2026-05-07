from collections import Counter
import argparse

import torch
from sklearn.model_selection import train_test_split

from model import NewsClassifier
from preprocess import prepare_data


def build_matrix(model, texts):
    return torch.stack([model._tfidf(text) for text in texts])


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


def train_model(X_train, y_train, epochs=300, lr=0.1):
    model = NewsClassifier(load_weights=False)
    fit_idf(model, X_train)

    X_tensor = build_matrix(model, X_train)

    label_to_id = {label: index for index, label in enumerate(model.classes)}
    y_tensor = torch.tensor([label_to_id[label] for label in y_train])

    counts = Counter(y_train)
    weights = torch.tensor(
        [len(y_train) / (len(model.classes) * counts[label]) for label in model.classes],
        dtype=torch.float32,
    )

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model.classifier(X_tensor)
        loss = loss_fn(logits, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


parser = argparse.ArgumentParser()
parser.add_argument("--final", action="store_true")
args = parser.parse_args()

# X, y = prepare_data("url_dataset_headlines.csv")
X, y = prepare_data("combined_urls.csv")

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
        "classifier": model.classifier.state_dict(),
    },
    "model.pt",
)

print("Saved model.pt")
