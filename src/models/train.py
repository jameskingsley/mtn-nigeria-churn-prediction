import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Metrics
from sklearn.metrics import roc_auc_score, classification_report

import mlflow
import mlflow.sklearn
from clearml import Task, OutputModel

# Initialize ClearML Task for Experiment & Model Tracking
task = Task.init(
    project_name="MTN Nigeria Churn Prediction", 
    task_name="Model_Tournament_and_Registration"
)

# BULLETPROOF FIX: Get the script's exact directory to construct absolute tracking paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "tracking", "mlflow.db"))
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
mlflow.set_experiment("churn_tournament")

def run_tournament():
    # BULLETPROOF FIX: Use absolute path mapping to reliably pinpoint the cleaned dataset
    data_path = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "data", "processed", "mtn_customer_churn_cleaned.csv"))
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {data_path}. Run the data cleaning steps first.")
        
    df = pd.read_csv(data_path)
    
    # Feature and Target Separation (Drop leaky/non-predictive columns)
    X = df.drop(columns=['customer_id', 'full_name', 'date_of_purchase', 'customer_churn_status', 'reasons_for_churn'])
    y = df['customer_churn_status'].apply(lambda x: 1 if x == 'yes' else 0)
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Stratified split to preserve churn proportions
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    # Define the Candidate Models
    candidate_models = {
        "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42),
        "Gradient_Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
        "Logistic_Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    }
    
    best_score = -1
    best_model_name = None
    best_pipeline = None
    
    # Tournament Execution Loop
    for name, model in candidate_models.items():
        # Wrap preprocessing and model into a single Pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        with mlflow.start_run(run_name=name):
            print(f"\nTraining {name}...")
            pipeline.fit(X_train, y_train)
            
            # Predict probabilities for ROC-AUC
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            y_pred = pipeline.predict(X_test)
            
            roc_auc = roc_auc_score(y_test, y_proba)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            print(f"{name} ROC-AUC: {roc_auc:.4f}")
            
            # Log metrics to MLflow
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("f1_churned", report['1']['f1-score'])
            mlflow.sklearn.log_model(pipeline, f"model_{name}")
            
            # Log metrics to ClearML Dashboard Live
            task.get_logger().report_single_value(name=f'{name}_ROC_AUC', value=roc_auc)
            task.get_logger().report_single_value(name=f'{name}_F1_Score', value=report['1']['f1-score'])
            
            # Track the winning model architecture
            if roc_auc > best_score:
                best_score = roc_auc
                best_model_name = name
                best_pipeline = pipeline

    print(f"\nTournament Winner: {best_model_name} with an ROC-AUC of {best_score:.4f}!")
    
    # Saving Winner Locally and Registering Directly to ClearML Model Registry
    local_model_path = os.path.join(SCRIPT_DIR, "best_churn_model.pkl")
    import joblib
    joblib.dump(best_pipeline, local_model_path)
    
    print("Uploading and Registering winning architecture to ClearML Registry...")
    # Bound seamlessly to the task context to prevent framework/argument mismatch
    output_model = OutputModel(
        task=task, 
        name="MTN_Nigeria_Best_Churn_Model", 
        framework="scikit-learn"
    )
    output_model.update_weights(weights_filename=local_model_path, auto_delete_file=False)
    
    print(f"Best Model [{best_model_name}] successfully locked down and registered in ClearML!")

if __name__ == "__main__":
    run_tournament()