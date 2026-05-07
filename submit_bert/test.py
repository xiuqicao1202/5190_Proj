from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from model import get_model
from preprocess import prepare_data
import numpy as np


X, y = prepare_data("url_with_headlines.csv")

# _, X_test, _, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y,
# )

y_int = [1 if label == "FoxNews" else 0 for label in y]
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=np.array(y_int)
)
y_temp_int = [1 if label == "FoxNews" else 0 for label in y_temp]
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=2 / 3, random_state=42, stratify=np.array(y_temp_int)
)

model = get_model()
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, pos_label="FoxNews")

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 score: {f1:.4f}")
print("Sample predictions:", predictions[:5])
