import streamlit as st
import requests

# Page setup matching corporate theme
st.set_page_config(page_title="MTN Nigeria Churn Gateway", page_icon="", layout="centered")

st.title("MTN Nigeria Churn Scoring Portal")
st.markdown("Enter customer behavior patterns to query the production machine learning model running on **FastAPI & ClearML**.")

# Setup Input Form Fields
with st.form("churn_input_form"):
    st.subheader("Customer Demographic & Account Info")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Customer Age", min_value=16, max_value=80, value=30)
        gender = st.selectbox("Gender", ["male", "female"])
        state = st.selectbox("State of Residence", ["osun", "abuja (fct)", "lagos", "enugu", "kano", "sokoto", "kwara"])
        mtn_device = st.selectbox("MTN Device Type", ["mobile sim card", "broadband mifi", "4g router", "5g broadband router"])
    
    with col2:
        customer_tenure_in_months = st.number_input("Customer Tenure (Months)", min_value=1, max_value=60, value=12)
        subscription_plan = st.selectbox("Data Plan", ["60gb monthly broadband plan", "12.5gb monthly plan", "150gb fup monthly unlimited", "1gb+1.5mins daily plan", "30gb monthly broadband plan"])
        satisfaction_rate = st.slider("Satisfaction Rating (1-5)", 1, 5, 3)
        customer_review = st.selectbox("Customer Review Category", ["poor", "fair", "good", "very good", "excellent"])

    st.subheader("Financial & Consumption Metrics")
    col3, col4 = st.columns(2)
    with col3:
        unit_price = st.number_input("Plan Unit Price (₦)", min_value=0, value=5000)
        number_of_times_purchased = st.number_input("Times Purchased in Month", min_value=1, value=2)
    with col4:
        data_usage = st.number_input("Actual Data Used (GB)", min_value=0.0, value=15.5)
        # Automatically calculate total revenue to match data sanity check constraints
        total_revenue = unit_price * number_of_times_purchased
        st.info(f"Calculated Total Revenue: ₦{total_revenue:,}")

    submit_btn = st.form_submit_with_value("Run Churn Risk Analysis")

# Package Payload & Fire API Post Request
if submit_btn:
    payload = {
        "age": int(age),
        "state": state,
        "mtn_device": mtn_device,
        "gender": gender,
        "satisfaction_rate": int(satisfaction_rate),
        "customer_review": customer_review,
        "customer_tenure_in_months": int(customer_tenure_in_months),
        "subscription_plan": subscription_plan,
        "unit_price": int(unit_price),
        "number_of_times_purchased": int(number_of_times_purchased),
        "total_revenue": int(total_revenue),
        "data_usage": float(data_usage)
    }
    
    try:
        # Route directly to the local FastAPI server
        api_url = "http://127.0.0.1:8000/predict"
        with st.spinner("Querying ClearML orchestration layers..."):
            response = requests.post(api_url, json=payload)
            
        if response.status_code == 200:
            result = response.json()
            prob = result["churn_probability"] * 100
            prediction = result["churn_prediction"]
            
            # Display Metric Panels
            st.write("---")
            st.subheader("Inference Outcome")
            if prediction == "yes":
                st.error(f"HIGH RISK OF CHURN: **{prob:.1f}% Churn Probability**")
            else:
                st.success(f"RETAINED CUSTOMER: **{prob:.1f}% Churn Probability**")
        else:
            st.error(f"API returned an error code: {response.status_code}")
            st.write(response.text)
            
    except requests.exceptions.ConnectionError:
        st.error("Connection Refused! Make sure your FastAPI app is running on port 8000 (`uvicorn src.main:app --reload`).")