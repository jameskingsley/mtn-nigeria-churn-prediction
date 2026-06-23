# 📱 MTN Nigeria Customer Churn Prediction Engine

An enterprise-grade, end-to-end MLOps production pipeline that trains, registers, hosts, and serves an optimized machine learning model to predict subscriber churn for MTN Nigeria. 

This system completely decouples code from heavy binary artifacts, utilizing a centralized cloud model registry for secure, authenticated artifact streaming at runtime.

---

## Live Infrastructure Links
* **Production API Gateway (FastAPI):** [https://mtn-nigeria-churn-prediction.onrender.com]
* **Interactive Scoring Portal (Streamlit):** [https://mtn-nigeria-churn-prediction-ez8fhrnajyyye9g46hmtct.streamlit.appR]

---

## System Architecture & MLOps Workflow

The system is split into three completely decoupled layers to maximize scalability, optimize system resources, and prevent repository bloat:

1. **Experimentation & Registration Layer (`src/models/train.py`):** Runs a localized, stratified model tournament across multiple candidate architectures (Random Forest, Gradient Boosting, Logistic Regression). The winning pipeline is versioned and its physical binary `.pkl` weight artifact is securely pushed up to ClearML Cloud Storage.
2. **Production Inference Layer (`src/main.py`):** A cloud-native FastAPI instance deployed on **Render**. On the initial bootstrap request, it securely authenticates via environment keys, downloads the active model weights from ClearML directly into memory cache via an authenticated stream, and processes high-throughput predictive queries.
3. **User Gateway Layer:** A lightweight Streamlit UI providing business stakeholders with an intuitive interface to query customer behavior patterns against the live cloud endpoint.

---

## Technology Stack
* **Core Language:** Python 
* **Modeling & Pipelines:** Scikit-Learn, Pandas, NumPy, Joblib
* **Experiment Tracking & Artifact Registry:** ClearML Cloud, MLflow (Local Tracking)
* **Application Layer:** FastAPI, Pydantic v2, Uvicorn, Gunicorn
* **Cloud Infrastructure & Deployment:** Render (Native Python Runtime)

---

## Repository Layout
```text
├── .gitignore
├── README.md
├── requirements.txt
└── src
    ├── main.py          
    └── models

##### API Specifications
* GET /
* Description: Service health check and target environment confirmation.

* Response Example:

 JSON
{
  "status": "healthy",
  "environment": "production_render",
  "registry": "clearml_cloud"
}

###### POST /predict
* Description: Accepts a structured customer behavioral payload and returns exact churn probabilities.

* Request Body (JSON):

JSON
{
  "age": 34,
  "state": "osun",
  "mtn_device": "mobile sim card",
  "gender": "male",
  "satisfaction_rate": 1,
  "customer_review": "poor",
  "customer_tenure_in_months": 12,
  "subscription_plan": "60gb monthly broadband plan",
  "unit_price": 10000,
  "number_of_times_purchased": 1,
  "total_revenue": 10000,
  "data_usage": 45.2
}
* Response Success (200 OK):

JSON
{
  "churn_probability": 0.241,
  "churn_prediction": "no"
}
###### Environment Variables & Configuration
To safely run the API gateway on Render or locally without configuration mismatches, the following environment keys must be bound to the application environment:

###### Code snippet
CLEARML_API_HOST=[https://api.clear.ml](https://api.clear.ml)
CLEARML_WEB_HOST=[https://app.clear.ml](https://app.clear.ml)
CLEARML_FILES_HOST=[https://files.clear.ml](https://files.clear.ml)
CLEARML_API_ACCESS_KEY=your_clearml_access_key_here
CLEARML_API_SECRET_KEY=your_clearml_secret_key_here
###### Local Development Setup
Clone the Repository:

Bash
git clone [https://github.com/your-username/mtn-nigeria-churn-prediction.git](https://github.com/your-username/mtn-nigeria-churn-prediction.git)
cd mtn-nigeria-churn-prediction
Initialize and Activate Virtual Environment:

Bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
Install Core Dependencies:

Bash
pip install -r requirements.txt
Execute Training Tournament:

Bash
python src/models/train.py
Run Local FastAPI Development Server:

Bash
uvicorn src.main:app --reload