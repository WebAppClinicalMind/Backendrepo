from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np

app = FastAPI(title="DataMining Service")

class DataFrameInput(BaseModel):
    data: dict  # expects {"columns": [...], "values": [[...],[...]]}
    target_cols: list

@app.post("/stats")
def relevant_features(input: DataFrameInput):
    df = pd.DataFrame(input.data["values"], columns=input.data["columns"])
    target_cols = input.target_cols

    # Only numeric columns
    numeric_df = df.select_dtypes(include=['int64','float64'])

    results = {}
    for target in target_cols:
        if target not in numeric_df.columns:
            continue
        # Correlation of all features with this target
        corr = numeric_df.corr()[target].drop(target).sort_values(key=abs, ascending=False)
        # Take top 5 most correlated features
        results[target] = corr.head(5).to_dict()

    return results
