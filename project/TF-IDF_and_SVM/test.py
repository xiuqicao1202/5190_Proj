import argparse
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from model import get_model
from preprocess import prepare_data


BASE_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=BASE_DIR / "combined_urls.csv")
args = parser.parse_args()

X, y = prepare_data(args.data)

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = get_model()
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="weighted")

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 score: {f1:.4f}")
print("Sample predictions:", predictions[:5])
