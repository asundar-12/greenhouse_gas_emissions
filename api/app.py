from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import predict_emissions

app = FastAPI(
    title = "Greenhouse Gas Emissions Predictor",
    description = "API for predicting combined Scope 1 and Scope 2 emissions",
    version = "0.1.0"
)

class EmissionsPredictionRequest(BaseModel):
    reporting_year: int = Field (example = 2026)
    scope1_mt_co2e: float = Field(example = 7.1)
    scope2_location_mt_co2e: float = Field(example = 0.5)
    emissions_intensity_mt_per_musd: float = Field(example = 0.5)
    revenue_usd_millions: float = Field (example = 16816.0)
    net_zero_target_year: float = Field (example = 2050)
    country: str = Field(example= "Australia")
    region: str = Field(example= "Asia-Pacific")
    sector: str = Field(example= "Energy")
    industry: str = Field(example= "E&P (LNG)")
    third_party_verified: bool = Field(example= True)
    
                                       
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: EmissionsPredictionRequest):
    input_data = request.model_dump()
    prediction = predict_emissions(input_data)
    return {"predicted_scope_1_plus_scope_2_location_mt": prediction}
    