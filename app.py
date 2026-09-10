import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

# Function to load models and scaler
@st.cache_resource
def load_resources():
    with open('combined_models.pkl', 'rb') as f:
        combined_models = pickle.load(f)
    with open('scaler_clf.pkl', 'rb') as f:
        scaler_clf = pickle.load(f)
    with open('label_encoder_clf.pkl', 'rb') as f:
        label_encoder_clf = pickle.load(f)
    return combined_models['classification_model'], combined_models['regression_model'], scaler_clf, label_encoder_clf

classification_model, regression_model, scaler_clf, label_encoder_clf = load_resources()

st.title('Supply Chain Risk and Delivery Time Prediction')
st.write('Enter the features to predict \'Risk Classification\' and \'Delivery Time Deviation\'.')

# Define input features based on the `x` dataframe columns
# Exclude 'timestamp' as it was dropped before scaling/training
input_features = [
    'vehicle_gps_latitude', 'vehicle_gps_longitude', 'fuel_consumption_rate',
    'eta_variation_hours', 'traffic_congestion_level', 'warehouse_inventory_level',
    'loading_unloading_time', 'handling_equipment_availability',
    'order_fulfillment_status', 'weather_condition_severity', 'port_congestion_level',
    'shipping_costs', 'supplier_reliability_score', 'lead_time_days',
    'historical_demand', 'iot_temperature', 'cargo_condition_status',
    'route_risk_level', 'customs_clearance_time', 'driver_behavior_score',
    'fatigue_monitoring_score', 'disruption_likelihood_score', 'delay_probability'
]

# Create input fields for each feature
user_inputs = {}
for feature in input_features:
    # Using a generic number input for simplicity; min/max/step could be refined based on data.describe()
    # Using default value from the original dataset's mean for better user experience.
    default_value = float(data[feature].mean()) if feature in data.columns else 0.0 # Access global 'data' DataFrame if available
    user_inputs[feature] = st.number_input(f'Enter {feature.replace("_", " ").title()}', value=default_value, format="%.4f")

if st.button('Predict'):
    # Convert user inputs to a DataFrame
    input_df = pd.DataFrame([user_inputs])

    # Scale the input features for classification using the loaded scaler
    scaled_input_clf = scaler_clf.transform(input_df)

    # Make classification prediction
    clf_prediction_encoded = classification_model.predict(scaled_input_clf)
    clf_prediction_label = label_encoder_clf.inverse_transform(clf_prediction_encoded)[0]

    # Make regression prediction (using the same scaled input for simplicity, though reg_scaler would be ideal)
    # If you had a separate scaler_reg, you'd use it here:
    # scaled_input_reg = scaler_reg.transform(input_df)
    # reg_prediction = regression_model.predict(scaled_input_reg)[0]
    # For now, using the classification scaler for regression prediction for demonstration, assuming features are the same.
    reg_prediction = regression_model.predict(scaled_input_clf)[0]

    st.subheader('Prediction Results:')
    st.write(f"**Risk Classification:** {clf_prediction_label}")
    st.write(f"**Predicted Delivery Time Deviation:** {reg_prediction:.2f} hours")

    st.subheader('Input Features:')
    st.write(input_df)
