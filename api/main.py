# main.py
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import numpy as np
from schema import ClienteInput, PrediccionOutput

# ── Carga del modelo ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global modelo
    modelo = joblib.load("modelo_churn.pkl")
    yield

app = FastAPI(
    title="API de predicción de churn",
    description="Predice la probabilidad de fuga de clientes de telecomunicaciones.",
    version="1.0.0",
    lifespan=lifespan
)

# ── Funciones auxiliares ──────────────────────────────────────────────────
def construir_features(cliente: dict) -> pd.DataFrame:
    df = pd.DataFrame([cliente])
    
    # Features derivadas — mismas que en el notebook
    servicios = ["OnlineSecurity","OnlineBackup","DeviceProtection",
                 "TechSupport","StreamingTV","StreamingMovies"]
    df["total_servicios"]    = df[servicios].apply(lambda x: (x == "Yes").sum(), axis=1)
    df["cargo_por_servicio"] = df["MonthlyCharges"] / (df["total_servicios"] + 1)
    df["riesgo_temprano"]    = (
        (df["Contract"] == "Month-to-month") & (df["tenure"] <= 12)
    ).astype(int)
    df["tenure_norm"] = df["tenure"] / 72
    
    return df

def clasificar_riesgo(prob: float) -> str:
    if prob >= 0.6:
        return "Alto"
    elif prob >= 0.4:
        return "Medio"
    return "Bajo"

# ── Endpoints ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "modelo": "logistic_regression_v1.0"}

@app.get("/metadata")
def metadata():
    return {
        "modelo"  : "Logistic Regression",
        "version" : "1.0.0",
        "metricas": {
            "recall"   : 0.80,
            "precision": 0.50,
            "roc_auc"  : 0.842
        },
        "features": 23,
        "threshold": 0.5
    }

@app.post("/predecir", response_model=PrediccionOutput)
def predecir(cliente: ClienteInput):
    try:
        df = construir_features(cliente.model_dump())
        prob  = modelo.predict_proba(df)[0][1]
        pred  = int(prob >= 0.5)
        return PrediccionOutput(
            churn_predicho=pred,
            probabilidad_churn=round(float(prob), 4),
            riesgo=clasificar_riesgo(prob)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predecir/lote")
def predecir_lote(clientes: list[ClienteInput]):
    try:
        resultados = []
        for cliente in clientes:
            df   = construir_features(cliente.model_dump())
            prob = modelo.predict_proba(df)[0][1]
            pred = int(prob >= 0.5)
            resultados.append({
                "churn_predicho"    : pred,
                "probabilidad_churn": round(float(prob), 4),
                "riesgo"            : clasificar_riesgo(prob)
            })
        return {"total": len(resultados), "predicciones": resultados}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))