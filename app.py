import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pandas as pd
import pickle

# load the trained model
model = tf.keras.models.load_model('model.h5')

# load the encoders and scaler
with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


# Streamlit app
st.title("Customer Churn Prediction")   

## user input
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18,92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

## 1. Prepare input data DataFrame (renamed from geo_encoded to input_data)
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],   
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# 2. One-hot encode 'Geography'
geo_encoded_array = onehot_encoder_geo.transform([[geography]]).toarray()

# 3. Create DataFrame from geo_encoded_array (FIXED: passed geo_encoded_array, not geo_encoded)
geo_encoded_df = pd.DataFrame(
    geo_encoded_array, 
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# 4. Combine one-hot encoded columns with input_data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# 5. Align input_data columns to match what the scaler was trained on
input_data = input_data.reindex(columns=scaler.feature_names_in_, fill_value=0)

# 6. Scale the input data
input_data_scaled = scaler.transform(input_data)

# 7. Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = float(prediction[0][0])

# 8. Display result in Streamlit
if prediction_proba > 0.5:
    st.write(f"⚠️ **The customer is likely to churn.** (Probability: {prediction_proba:.2%})")
else:
    st.write(f"✅ **The customer is not likely to churn.** (Probability: {prediction_proba:.2%})")