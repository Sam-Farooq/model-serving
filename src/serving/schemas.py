from pydantic import BaseModel, Field


class Features(BaseModel):
    amount: float = Field(ge=0)
    hour_of_day: int = Field(ge=0, le=23)
    merchant_risk: float = Field(ge=0, le=1)
    account_age_days: int = Field(ge=0)
    txn_count_24h: int = Field(ge=0)
    distinct_countries_7d: int = Field(ge=0)
    is_cross_border: bool = False

    def to_vector(self) -> list[float]:
        # Order is the contract between this file and the trained model. It is
        # asserted in tests against the artifact's recorded feature_order.
        return [
            self.amount,
            float(self.hour_of_day),
            self.merchant_risk,
            float(self.account_age_days),
            float(self.txn_count_24h),
            float(self.distinct_countries_7d),
            float(self.is_cross_border),
        ]


FEATURE_ORDER = [
    "amount", "hour_of_day", "merchant_risk", "account_age_days",
    "txn_count_24h", "distinct_countries_7d", "is_cross_border",
]


class PredictRequest(BaseModel):
    instances: list[Features] = Field(min_length=1, max_length=512)


class Prediction(BaseModel):
    score: float
    label: bool


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    model_version: str
    served_by: str          # "inline" or "celery"
    latency_ms: float
