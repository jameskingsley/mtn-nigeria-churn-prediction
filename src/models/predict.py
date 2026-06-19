import os
import joblib
import pandas as pd
from clearml import Model

# Global variable to cache the model pipeline so we don't fetch from cloud on every API call
_CACHED_MODEL = None

def load_clearml_model():
    """
    Dynamically pulls the latest registered production weights 
    from the MTN Nigeria Churn Prediction registry inside your ClearML workspace.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        print("Fetching winning Gradient Boosting model weights from ClearML Registry...")
        
        # Query  registered model by name and project context
        clearml_model = Model.query_models(
            project_name="MTN Nigeria Churn Prediction",
            model_name="MTN_Nigeria_Best_Churn_Model",
            only_published=False  # Set to True if explicitly locked to published states
        )[0]
        
        # Download weights locally to a temp path managed by ClearML cache
        local_weight_path = clearml_model.get_local_copy()
        
        # Deserialise the scikit-learn Pipeline
        _CACHED_MODEL = joblib.load(local_weight_path)
        print("Production weights loaded and cached successfully!")
        
    return _CACHED_MODEL

def get_prediction(input_data: dict) -> dict:
    """
    Exposes live predictions using the model fetched from ClearML.
    """
    model = load_clearml_model()
    
    # Structure features into a single row DataFrame
    df_input = pd.DataFrame([input_data])
    
    proba = model.predict_proba(df_input)[0][1]
    prediction = model.predict(df_input)[0]
    
    return {
        "churn_probability": float(proba),
        "churn_prediction": "yes" if prediction == 1 else "no"
    }