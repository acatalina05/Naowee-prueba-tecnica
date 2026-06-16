# schema.py
from pydantic import BaseModel
from typing import Literal

class ClienteInput(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No"]
    OnlineBackup: Literal["Yes", "No"]
    DeviceProtection: Literal["Yes", "No"]
    TechSupport: Literal["Yes", "No"]
    StreamingTV: Literal["Yes", "No"]
    StreamingMovies: Literal["Yes", "No"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float
    TotalCharges: float

class PrediccionOutput(BaseModel):
    churn_predicho: int
    probabilidad_churn: float
    riesgo: Literal["Alto", "Medio", "Bajo"]