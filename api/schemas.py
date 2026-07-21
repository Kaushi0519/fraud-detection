from datetime import datetime

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    cc_num: int = Field(..., description="Card/user identifier")
    amt: float = Field(..., gt=0, description="Transaction amount in USD")
    category: str = Field(..., description="Merchant category, e.g. 'grocery_pos'")
    lat: float = Field(..., description="Customer latitude")
    long: float = Field(..., description="Customer longitude")
    merch_lat: float = Field(..., description="Merchant latitude")
    merch_long: float = Field(..., description="Merchant longitude")
    trans_date_trans_time: datetime = Field(..., description="Transaction timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "cc_num": 60416207185,
                "amt": 120.50,
                "category": "shopping_net",
                "lat": 36.0788,
                "long": -81.1781,
                "merch_lat": 36.430124,
                "merch_long": -81.179483,
                "trans_date_trans_time": "2020-06-21T14:32:00",
            }
        }
    }


class FeatureContribution(BaseModel):
    feature: str
    value: float
    shap_contribution: float


class PredictionResponse(BaseModel):
    fraud_score: float = Field(..., description="Predicted probability of fraud, 0-1")
    is_fraud: bool = Field(..., description="fraud_score thresholded at 0.5")
    top_features: list[FeatureContribution] = Field(
        ..., description="Features that most influenced this prediction"
    )
