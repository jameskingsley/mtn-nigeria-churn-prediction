import os
import sys
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from clearml import Model
from clearml.storage import StorageManager  
# Ensure smooth absolute package routing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(
    title="MTN Nigeria Churn Production API",
    description="Cloud-deployed inference engine backed by cloud-hosted ClearML artifacts.",
    version="1.0.0"
)

# Enable CORS so the Streamlit portal can query the Render endpoint securely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to cache the weights in memory once downloaded
_PRODUCTION_MODEL = None

def download_and_cache_clearml_model():
    global _PRODUCTION_MODEL
    if _PRODUCTION_MODEL is None:
        try:
            print("Production Boot: Fetching latest model metadata from ClearML...")
            # Query the precise published model from your workspace
            clearml_model = Model.query_models(
                project_name="MTN Nigeria Churn Prediction",
                model_name="MTN_Nigeria_Best_Churn_Model",
                only_published=True
            )[0]
            
            # Extract the raw cloud url (https://files.clear.ml/...)
            model_url = clearml_model.url
            if not model_url or not model_url.startswith("https"):
                raise ValueError(f"Extracted an invalid cloud URL from registry: {model_url}")
                
            print(f"🔗 Authenticated cloud link discovered: {model_url}")
            
            # Use ClearML's StorageManager to download the secure link.
            print("Downloading weights securely via ClearML Storage Manager...")
            local_path = StorageManager.get_local_copy(remote_url=model_url)
            
            print(f"Verified artifact weight target resolved to: {local_path}")
            
            # Load the scikit-learn metrics pipeline into active system RAM
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
    return {"status": "healthy", "environment": "production_render", "registry": "clearml_cloud"}

@app.post("/predict")
def predict_churn(payload: CustomerFeatures):
    try:
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