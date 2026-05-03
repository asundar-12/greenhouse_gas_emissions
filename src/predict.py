from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/model.pkl")


def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    return joblib.load(path)

def create_input_dataframe(input_data: dict) -> pd.DataFrame:
    return pd.DataFrame([input_data])

def predict_emissions(input_data: dict) -> float:
    model = load_model()
    input_df = create_input_dataframe(input_data)
    prediction = model.predict(input_df)
    return float(prediction[0])

if __name__ == "__main__":
    sample_input = {
        "reporting_year": 2023,
        "scope1_mt_co2e": 7.1,
        "scope2_location_mt_co2e": 0.5,
        "emissions_intensity_mt_per_musd": 0.000452,
        "revenue_usd_millions": 16816.0,
        "net_zero_target_year": 2050,
        "country": "Australia",
        "region": "Asia-Pacific",
        "sector": "Energy",
        "industry": "E&P (LNG)",
        "third_party_verified": True
    }
    predicted_emissions = predict_emissions(sample_input)

    print(f"Predicted Scope 1 + Scope 2 emissions: {predicted_emissions:.4f}")