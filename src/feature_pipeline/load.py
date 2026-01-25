"""
Load and time-split the raw dataset.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")


def load_and_split_data(
    raw_path: str = "data/raw/raw_housing_data.csv", output_dir: Path | str = DATA_DIR
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Load the dataset
    df = pd.read_csv(raw_path)

    # Convert 'date' column to datetime
    df["date"] = pd.to_datetime(df["date"])
    # Sort by date
    df = df.sort_values(by="date")

    # Define the split date
    cutoff_date_test = pd.Timestamp("2020-01-01")
    cutoff_date_holdout = pd.Timestamp("2022-01-01")

    # Split the dataset
    train_df = df[df["date"] < cutoff_date_test]
    test_df = df[(df["date"] >= cutoff_date_test) & (df["date"] < cutoff_date_holdout)]
    holdout_df = df[df["date"] >= cutoff_date_holdout]

    # Save the splits
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(outdir / "train_data.csv", index=False)
    test_df.to_csv(outdir / "test_data.csv", index=False)
    holdout_df.to_csv(outdir / "holdout_data.csv", index=False)

    print(
        f"   Train: {train_df.shape}, Test: {test_df.shape}, Holdout: {holdout_df.shape}"
    )

    return train_df, test_df, holdout_df


if __name__ == "__main__":
    load_and_split_data()
