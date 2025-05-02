# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("bosch_pricing.csv")
    return df

df = load_data()

st.title("Bosch Product Pricing Optimization")
st.markdown("Analyze demand elasticity and optimize pricing based on customer preferences and competitor pricing.")

# Display raw data
if st.checkbox("Show Raw Data"):
    st.write(df)

# Feature Engineering
df['price_diff'] = df['base_price'] - df['competitor_price']

# Modeling
features = ['base_price', 'competitor_price', 'advertising_spend', 'seasonality_index', 'customer_segment', 'category']
target = 'demand_volume'

X = df[features]
y = df[target]

categorical = ['customer_segment', 'category']
numerical = ['base_price', 'competitor_price', 'advertising_spend', 'seasonality_index']

# Pipeline
preprocessor = ColumnTransformer([
    ('num', 'passthrough', numerical),
    ('cat', OneHotEncoder(), categorical)
])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model.fit(X, y)

# Save model
joblib.dump(model, 'pricing_model.pkl')

# Predict
st.subheader("Try Your Own Pricing Strategy")
input_base_price = st.slider("Base Price", min_value=40.0, max_value=500.0, step=1.0, value=250.0)
input_competitor_price = st.slider("Competitor Price", min_value=40.0, max_value=500.0, step=1.0, value=260.0)
input_ad_spend = st.slider("Advertising Spend", min_value=1000.0, max_value=10000.0, step=100.0, value=5000.0)
input_seasonality = st.slider("Seasonality Index", min_value=0.7, max_value=1.3, step=0.01, value=1.0)
input_segment = st.selectbox("Customer Segment", options=df['customer_segment'].unique())
input_category = st.selectbox("Category", options=df['category'].unique())

# Create input DataFrame
input_data = pd.DataFrame([{
    'base_price': input_base_price,
    'competitor_price': input_competitor_price,
    'advertising_spend': input_ad_spend,
    'seasonality_index': input_seasonality,
    'customer_segment': input_segment,
    'category': input_category
}])

predicted_demand = model.predict(input_data)[0]
st.success(f"Predicted Demand Volume: {int(predicted_demand)} units")

# Elasticity Estimation (approximate)
st.subheader("Price Elasticity Estimate")
delta_price = 1.0  # $1 change
elasticity_input = input_data.copy()
elasticity_input['base_price'] += delta_price
demand_up = model.predict(elasticity_input)[0]
elasticity = ((demand_up - predicted_demand) / predicted_demand) / (delta_price / input_base_price)

st.info(f"Elasticity (ΔDemand/ΔPrice): {elasticity:.2f}")
if elasticity < -1:
    st.warning("Highly elastic: consider lowering price.")
elif -1 <= elasticity < 0:
    st.success("Inelastic: price increases may be profitable.")
else:
    st.error("Demand increases with price — likely overfitting or special case.")

