import streamlit as st
import numpy as np  
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle


#Load the trained model
model = tf.keras.models.load_model('model.h5')

#Load the scaler and encoder
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)  

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)      

##streamlit app
## Streamlit app
st.title("Customer Churn Prediction")    

# User inputs
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.number_input('Age', min_value=18, max_value=100, value=30)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', min_value=0, max_value=10)
num_of_products = st.slider('Number of Products', min_value=1, max_value=4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data with correct column names
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],  # Correct capitalization
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Reorder columns to match scaler feature names
input_data = input_data[scaler.feature_names_in_]

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict Churn
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

if prediction_prob > 0.5:
    st.write(f'The customer is likely to churn with a probability of {prediction_prob:.2f}')
else:
    st.write(f'The customer is unlikely to churn with a probability of {1 - prediction_prob:.2f}')
