from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("./data/raw/global_corporate_ghg_emissions_2022_2023.csv")
PROCESSED_DATA_PATH = Path("./data/processed/modeling_dataset.csv")


REQUIRED_COLUMNS = [
    "company_id",
    "company_name",
    "country",
    "region",
    "sector",
    "industry",
    "reporting_year",
    "scope1_mt_co2e",
    "scope2_location_mt_co2e",
    "scope1_plus_scope2_location_mt",
    "emissions_intensity_mt_per_musd",
    "revenue_usd_millions",
    "third_party_verified",
    "net_zero_target_year",
]


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw greenhouse gas emissions dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame) -> None:
    """Ensure the raw dataset contains the columns required by the pipeline."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data and return a modeling-ready dataframe."""
    df = df.copy()

    modeling_columns = [
        "company_id",
        "company_name",
        "country",
        "region",
        "sector",
        "industry",
        "reporting_year",
        "scope1_mt_co2e",
        "scope2_location_mt_co2e",
        "scope1_plus_scope2_location_mt",
        "emissions_intensity_mt_per_musd",
        "revenue_usd_millions",
        "third_party_verified",
        "net_zero_target_year",
    ]

    df = df[modeling_columns]

    numeric_columns = [
        "scope1_mt_co2e",
        "scope2_location_mt_co2e",
        "scope1_plus_scope2_location_mt",
        "emissions_intensity_mt_per_musd",
        "revenue_usd_millions",
        "net_zero_target_year",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["third_party_verified"] = df["third_party_verified"].astype(bool)

    df["scope2_location_mt_co2e"] = df["scope2_location_mt_co2e"].fillna(0)
    df["net_zero_target_year"] = df["net_zero_target_year"].fillna(0)

    df = df.dropna(
        subset=[
            "scope1_plus_scope2_location_mt",
            "emissions_intensity_mt_per_musd",
            "revenue_usd_millions",
        ]
    )

    return df


def save_processed_data(
    df: pd.DataFrame,
    path: Path = PROCESSED_DATA_PATH,
) -> None:
    """Save processed data to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_data_processing() -> pd.DataFrame:
    """Run the full data processing step."""
    raw_df = load_raw_data()
    validate_columns(raw_df)

    processed_df = clean_data(raw_df)
    save_processed_data(processed_df)

    return processed_df


if __name__ == "__main__":
    processed_data = run_data_processing()

    print(f"Processed data saved to: {PROCESSED_DATA_PATH}")
    print(f"Rows: {processed_data.shape[0]}")
    print(f"Columns: {processed_data.shape[1]}")
