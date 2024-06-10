from typing import Union, List, Dict
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import mlflow
import numpy as np
import pandas as pd
import sklearn
import os
import uvicorn
import shap

# Define the input data model
class InputData(BaseModel):
    dataframe_records: List[Dict[str, float]]

app = FastAPI()


# Construct the absolute path to the MLflow model
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "production_model")
model = mlflow.sklearn.load_model(model_path)
explainer = shap.Explainer(model)

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Define the prediction endpoint
@app.post("/predict")
async def predict(input_data: InputData): # we could use InputData with a Pydantic model based on mlflow signature
    try:
        # Convert input data to DataFrame
        data_df = pd.DataFrame(input_data.dataframe_records)

        # Get prediction from the model
        prediction = model.predict_proba(data_df)

        # Return the prediction as a response
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain_local")
async def shap_values(input_data: InputData): # we could use InputData with a Pydantic model based on mlflow signature
    try:
        # Convert input data to DataFrame
        data_df = pd.DataFrame(input_data.dataframe_records)

        # Get prediction from the model
        shap_values = explainer(data_df)

        shap_values_response = {
            "base_values": shap_values.base_values.tolist(),
            "values": shap_values.values.tolist(),
            "data": shap_values.data.tolist(),
            "feature_names": data_df.columns.tolist()
        }

        # Return the SHAP values as a response
        return shap_values_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run('myapp:app', host='0.0.0.0', port=8000)