from preprocess import prepare_data
import pandas as pd

SOURCE_CSV = "training_data.csv"
OUTPUT_CSV = "training_data_preprocessed.csv"

# Read from training_data.csv and preprocess it
X, y = prepare_data(SOURCE_CSV)

# Save results in machine-friendly CSV format
result_df = pd.DataFrame(
    {
        "processed_title": X,
        "label": y,
    }
)
result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print(f"Processed {len(result_df)} rows from {SOURCE_CSV}")
print(f"Saved preprocessed data to {OUTPUT_CSV}")