from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

app = FastAPI(title="Classification Service")

class ClassificationInput(BaseModel):
    data: dict  # {"columns": [...], "values": [[...],[...]]}
    target_cols: list

@app.post("/classify")
def classify(input: ClassificationInput):
    df = pd.DataFrame(input.data["values"], columns=input.data["columns"])
    target_cols = input.target_cols

    # Features and targets
    X = df.drop(columns=target_cols, errors='ignore')
    y = df[target_cols]

    # Scale numeric features
    numeric_cols = X.select_dtypes(include=['int64','float64']).columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # SMOTE for each target separately (simplified: apply to first target only for demo)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y[target_cols[0]])

    # Train RandomForest for first label
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf.fit(X_res, y_res)

    # Predict for all data (demo: using same X)
    y_pred = rf.predict(X)
    proba = rf.predict_proba(X)[:,1]

    # Threshold adjustment (simple)
    threshold = 0.3
    y_pred_binary = (proba > threshold).astype(int)

    results = {
        "predictions": y_pred_binary.tolist(),
        "probabilities": proba.tolist(),
        "threshold": threshold
    }
    return results
