import os
import sys
import joblib
import urllib.request
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from clearml import Model

# Ensure smooth absolute package routing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(
    title="MTN Nigeria Churn Production API",
    description="Cloud-deployed inference engine backed by ClearML model artifacts.",
    version="1.0.0"
)

# Enable CORS so Streamlit app can query the Render endpoint securely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to cache the weights in memory on Render's container instance
_PRODUCTION_MODEL = None

def download_and_cache_clearml_model():
    global _PRODUCTION_MODEL
    if _PRODUCTION_MODEL is None:
        try:
            print(" Production Boot: Querying model configuration metadata from ClearML...")
            # Query the precise model configuration from the workspace
            clearml_model = Model.query_models(
                project_name="MTN Nigeria Churn Prediction",
                model_name="MTN_Nigeria_Best_Churn_Model",
                only_published=False
            )[0]
            
            # BYPASS: Extract the absolute cloud storage link directly from ClearML's backend registry
            model_url = clearml_model.url
            if not model_url:
                raise ValueError("Could not extract a valid storage URL from the ClearML model object.")
                
            print(f"Direct artifact link discovered: {model_url}")
            
            # Explicitly force the target file storage path straight to Render's writable temp directory
            local_path = "/tmp/best_churn_model.pkl"
            
            # Download the file directly bypassing ClearML's internal caching client completely
            print(f Downloading weights directly into: {local_path} ...")
            urllib.request.urlretrieve(model_url, local_path)
            
            print(f"Verified artifact weight target resolved to: {local_path}")
            
            # Unpickle scikit-learn metrics pipeline into RAM memory
            _PRODUCTION_MODEL = joblib.load(local_path)
            print("Successfully anchored model into memory cache!")
        except Exception as e:
            print(f"Critical Initialisation Error: {str(e)}")
            raise e
    return _PRODUCTION_MODEL

class CustomerFeatures(BaseModel):
    age: int
    state: str
    mtn_device: str
    gender: str
    satisfaction_rate: int
    customer_review: str
    customer_tenure_in_months: int
    subscription_plan: str
    unit_price: int
    number_of_times_purchased: int
    total_revenue: int
    data_usage: float

@app.get("/")
def health_check():
    return {"status": "healthy", "environment": "production_render", "registry": "clearml"}

@app.post("/predict")
def predict_churn(payload: CustomerFeatures):
    try:
        # Pulls from ClearML URL on the 1st request, hits RAM cache on all following requests
        model = download_and_cache_clearml_model()
        input_df = pd.DataFrame([payload.model_dump()])
        
        # Run cloud scoring execution
        proba = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]
        
        return {
            "churn_probability": float(proba),
            "churn_prediction": "yes" if prediction == 1 else "no"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Failure: {str(e)}")