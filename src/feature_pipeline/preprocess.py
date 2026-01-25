"""
Data preprocessing module for the feature pipeline.
This module includes functions for cleaning, transforming, and preparing data for feature extraction.
"""

import pandas as pd
import re
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Previous city mapping for non-standard city names
CITY_MAPPING = {
    "Las Vegas-Henderson-Paradise": "Las Vegas-Henderson-North Las Vegas, NV",
    "Denver-Aurora-Lakewood": "Denver-Aurora-Centennial, CO",
    "Houston-The Woodlands-Sugar Land": "Houston-Pasadena-The Woodlands, TX",
    "Austin-Round Rock-Georgetown": "Austin-Round Rock-San Marcos, TX",
    "Miami-Fort Lauderdale-Pompano Beach": "Miami-Fort Lauderdale-West Palm Beach, FL",
    "San Francisco-Oakland-Berkeley": "San Francisco-Oakland-Fremont, CA",
    "DC_Metro": "Washington-Arlington-Alexandria, DC-VA-MD-WV",
    "Atlanta-Sandy Springs-Alpharetta": "Atlanta-Sandy Springs-Roswell, GA",
    "Pittsburgh": "Pittsburgh, PA",
    "Boston-Cambridge-Newton": "Boston-Cambridge-Newton, MA-NH",
    "Tampa-St. Petersburg-Clearwater": "Tampa-St. Petersburg-Clearwater, FL",
    "Baltimore-Columbia-Towson": "Baltimore-Columbia-Towson, MD",
    "Portland-Vancouver-Hillsboro": "Portland-Vancouver-Hillsboro, OR-WA",
    "Philadelphia-Camden-Wilmington": "Philadelphia-Camden-Wilmington, PA-NJ-DE-MD",
    "New York-Newark-Jersey City": "New York-Newark-Jersey City, NY-NJ",
    "Chicago-Naperville-Elgin": "Chicago-Naperville-Elgin, IL-IN",
    "Orlando-Kissimmee-Sanford": "Orlando-Kissimmee-Sanford, FL",
    "Seattle-Tacoma-Bellevue": "Seattle-Tacoma-Bellevue, WA",
    "San Diego-Chula Vista-Carlsbad": "San Diego-Chula Vista-Carlsbad, CA",
    "St. Louis": "St. Louis, MO-IL",
    "Sacramento-Roseville-Folsom": "Sacramento-Roseville-Folsom, CA",
    "Phoenix-Mesa-Chandler": "Phoenix-Mesa-Chandler, AZ",
    "Riverside-San Bernardino-Ontario": "Riverside-San Bernardino-Ontario, CA",
    "San Antonio-New Braunfels": "San Antonio-New Braunfels, TX",
    "Detroit-Warren-Dearborn": "Detroit-Warren-Dearborn, MI",
    "Cincinnati": "Cincinnati, OH-KY-IN",
    "Charlotte-Concord-Gastonia": "Charlotte-Concord-Gastonia, NC-SC",
    "Los Angeles-Long Beach-Anaheim": "Los Angeles-Long Beach-Anaheim, CA",
    "Dallas-Fort Worth-Arlington": "Dallas-Fort Worth-Arlington, TX",
    "Minneapolis-St. Paul-Bloomington": "Minneapolis-St. Paul-Bloomington, MN-WI",
}


def normalize_city_name(city: str) -> str:
    """Lowercase, remove training spaces and standadize dashes"""
    if pd.isna(city):
        return city

    city = str(city).lower().strip()
    city = re.sub(r"\s*[-–—]\s*", "-", city)
    city = re.sub(r"\s+", " ", city)

    return city


def clean_and_merge_city_names(
    df: pd.DataFrame, metros_path: str | None = "data/raw/raw_usmetros.csv"
) -> pd.DataFrame:

    # Check if 'city_full' columns exists and proceed to replacement with mapping
    if "city_full" not in df.columns:
        print("'city_full' column not found in DataFrame.")
        return df

    # Replace non-standard city names using the mapping
    df["city_full"] = df["city_full"].replace(CITY_MAPPING)

    # If lat/lng columns already exist, skip merging
    if {"lat", "lng"}.issubset(df.columns):
        print
        return df

    # If the metros file is not provided or doesn't exist, skip merging
    if metros_path is None or not Path(metros_path).exists():
        print("Metros file not found. Skipping city name merging.")
        return df

    # Load and merge metros data
    metros_df = pd.read_csv(metros_path)
    metros_df["metro_full"] = metros_df["metro_full"].apply(normalize_city_name)
    df = df.merge(
        metros_df[["metro_full", "lat", "lng"]],
        left_on="city_full",
        right_on="metro_full",
        how="left",
    )
    df.drop(columns=["metro_full"], inplace=True)

    # Check for unmatched cities
    unmatched_cities = df[df["lat"].isnull()]["city_full"].unique()
    if len(unmatched_cities) > 0:
        print(f"Unmatched cities: {unmatched_cities}")
    else:
        print("All city names matched successfully.")

    return df


def remove_duplicatates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows excluding date and year columns.
    """
    initial_count = df.shape[0]
    df = df.drop_duplicates(subset=df.columns.difference(["date", "year"]), keep=False)
    final_count = df.shape[0]
    print(f"Removed {initial_count - final_count} duplicate rows.")

    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outliers with median_list_price above 19M.
    """
    initial_count = df.shape[0]
    df = df[df["median_list_price"] <= 19000000].copy()
    final_count = df.shape[0]
    print(f"Removed {initial_count - final_count} outliers.")

    return df


def preprocess_data_split(
    split: str,
    raw_dir: Path | str = RAW_DATA_DIR,
    processed_dir: Path | str = PROCESSED_DATA_DIR,
    metros_path: str | None = "data/raw/raw_usmetros.csv",
) -> pd.DataFrame:
    """
    Run preprocessing steps and save into processed directory.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    path = raw_dir / f"{split}_data.csv"
    df = pd.read_csv(path)

    df = clean_and_merge_city_names(df, metros_path=metros_path)
    df = remove_duplicatates(df)
    df = remove_outliers(df)

    out_path = processed_dir / f"cleaning_{split}_data.csv"
    df.to_csv(out_path, index=False)

    print(f"Preprocessed {split} data saved to {out_path}")
    return df


def run_preprocessing_pipeline(
    splits: tuple[str, ...] = ("train", "test", "holdout"),
    raw_dir: Path | str = RAW_DATA_DIR,
    processed_dir: Path | str = PROCESSED_DATA_DIR,
    metros_path: str | None = "data/raw/raw_usmetros.csv",
) -> None:
    """
    Run preprocessing for all specified data splits.
    """
    for split in splits:
        preprocess_data_split(
            split, raw_dir=raw_dir, processed_dir=processed_dir, metros_path=metros_path
        )


if __name__ == "__main__":
    run_preprocessing_pipeline()
