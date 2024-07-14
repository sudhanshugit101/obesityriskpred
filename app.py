import numpy as np
import pickle
import streamlit as st
import xgboost as xgb

# Load the pickled model
model = xgb.Booster()
model.load_model('model.xgb')

st.title('Obesity Prediction')

# User inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', min_value=0, max_value=120, value=0)
    height = st.number_input('Height (in cm)', min_value=0, max_value=300, value=0)
    weight = st.number_input('Weight (in kg)', min_value=0, max_value=300, value=0)
    fcvc = st.number_input('Frequency of Consumption of Vegetables (FCVC)', min_value=0, max_value=5, value=0)
    ncp = st.number_input('Number of Main Meals (NCP)', min_value=0, max_value=10, value=0)
    ch2o = st.number_input('Consumption of Water Daily (CH2O)', min_value=0, max_value=5, value=0)
    tue = st.number_input('Time Using Technology Devices (TUE)', min_value=0, max_value=24, value=0)

with col2:
    gender = st.selectbox('Gender', ['Female', 'Male'])
    family_history = st.selectbox('Family History with Overweight', ['Yes', 'No'])
    favc = st.selectbox('Frequent Consumption of High Caloric Food (FAVC)', ['Yes', 'No'])
    caec = st.selectbox('Consumption of Food between Meals (CAEC)', ['Always', 'Frequently', 'Sometimes', 'No'])
    smoke = st.selectbox('Smoking Habit (SMOKE)', ['Yes', 'No'])
    scc = st.selectbox('Monitor Caloric Intake (SCC)', ['Yes', 'No'])
    calc = st.selectbox('Consumption of Alcohol (CALC)', ['Always', 'Frequently', 'Sometimes', 'No'])
    mtrans = st.selectbox('Transportation used (MTRANS)', ['Automobile', 'Bike', 'Motorbike', 'Public Transportation', 'Walking'])

# Preprocessing
data = {
    'Age': age,
    'Height': height,
    'Weight': weight,
    'FCVC': fcvc,
    'NCP': ncp,
    'CH2O': ch2o,
    'TUE': tue,
    'Gender_Female': int(gender == 'Female'),
    'Gender_Male': int(gender == 'Male'),
    'family_history_with_overweight_no': int(family_history == 'No'),
    'family_history_with_overweight_yes': int(family_history == 'Yes'),
    'FAVC_no': int(favc == 'No'),
    'FAVC_yes': int(favc == 'Yes'),
    'CAEC_Always': int(caec == 'Always'),
    'CAEC_Frequently': int(caec == 'Frequently'),
    'CAEC_Sometimes': int(caec == 'Sometimes'),
    'CAEC_no': int(caec == 'No'),
    'SMOKE_no': int(smoke == 'No'),
    'SMOKE_yes': int(smoke == 'Yes'),
    'SCC_no': int(scc == 'No'),
    'SCC_yes': int(scc == 'Yes'),
    'CALC_Always': int(calc == 'Always'),
    'CALC_Frequently': int(calc == 'Frequently'),
    'CALC_Sometimes': int(calc == 'Sometimes'),
    'CALC_no': int(calc == 'No'),
    'MTRANS_Automobile': int(mtrans == 'Automobile'),
    'MTRANS_Bike': int(mtrans == 'Bike'),
    'MTRANS_Motorbike': int(mtrans == 'Motorbike'),
    'MTRANS_Public_Transportation': int(mtrans == 'Public Transportation'),
    'MTRANS_Walking': int(mtrans == 'Walking')
}

input_data = np.array([list(data.values())])

# Convert input data to DMatrix
dinput = xgb.DMatrix(input_data)

if st.button('Predict'):
    try:
        prediction = model.predict(dinput)
        obesity_types = ['Normal Weight', 'Obese', 'Overweight', 'Underweight']
        st.success(f'You are {obesity_types[np.argmax(prediction)]}')
    except ValueError as e:
        st.error(f'Error in prediction: {e}')
