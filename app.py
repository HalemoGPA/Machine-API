from fastapi.encoders import jsonable_encoder
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PowerTransformer, StandardScaler, OrdinalEncoder
from imblearn.over_sampling import ADASYN

from sklearn.ensemble import GradientBoostingClassifier

app = FastAPI()


model = joblib.load("model.pkl")
X = joblib.load("myX.pkl")


class InputData(BaseModel):
    gender: str
    age: float
    ever_married: str
    heart_disease: int
    hypertension: int
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str
    smoking_not_found: bool


@app.post("/predict/")
async def predict(data: InputData):
    # Create DataFrame from input data
    power_tranform_t, num_scaler_t, cat_scaler_t, ordinal_encoder_t = joblib.load(
        "transformers_prime.pkl"
    )
    df = {
        "gender": [data.gender],
        "age": [data.age],
        "hypertension": [data.hypertension],
        "heart_disease": [data.heart_disease],
        "ever_married": [data.ever_married],
        "work_type": [data.work_type],
        "Residence_type": [data.Residence_type],
        "avg_glucose_level": [data.avg_glucose_level],
        "bmi": [data.bmi],
        "smoking_status": [data.smoking_status],
        "smoking_not_found": [data.smoking_not_found],
    }

    X = joblib.load("myX.pkl")
    # return df
    df = pd.DataFrame(df, index=[0])
    df[["age", "avg_glucose_level", "bmi"]] = power_tranform_t.transform(
        df[["age", "avg_glucose_level", "bmi"]]
    )
    df[["age", "avg_glucose_level", "bmi"]] = num_scaler_t.transform(
        df[["age", "avg_glucose_level", "bmi"]]
    )
    df[
        [
            "gender",
            "hypertension",
            "heart_disease",
            "ever_married",
            "work_type",
            "Residence_type",
            "smoking_status",
            "smoking_not_found",
        ]
    ] = ordinal_encoder_t.transform(
        df[
            [
                "gender",
                "hypertension",
                "heart_disease",
                "ever_married",
                "work_type",
                "Residence_type",
                "smoking_status",
                "smoking_not_found",
            ]
        ]
    )
    df[
        [
            "gender",
            "hypertension",
            "heart_disease",
            "ever_married",
            "work_type",
            "Residence_type",
            "smoking_status",
            "smoking_not_found",
        ]
    ] = cat_scaler_t.transform(
        df[
            [
                "gender",
                "hypertension",
                "heart_disease",
                "ever_married",
                "work_type",
                "Residence_type",
                "smoking_status",
                "smoking_not_found",
            ]
        ]
    )

    prediction = model.predict(df)

    # Return prediction
    return {"prediction": int(prediction)}


@app.get("/")
async def main():
    return {"message": "Hello, World"}
