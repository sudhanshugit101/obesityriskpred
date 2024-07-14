import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('train.csv')  # Replace with your dataset path
    return data

# Manually preprocess and train the model
@st.cache_resource
def train_model():
    data = load_data()

    # Separate features and target
    X = data.drop(['NObeyesdad'], axis=1)
    y = data['NObeyesdad']

    # Manually preprocess numerical features
    numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    scaler = StandardScaler().fit(X[numerical_features])
    X[numerical_features] = scaler.transform(X[numerical_features])

    # Manually preprocess categorical features
    categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Save the trained model and preprocessing parameters
    joblib.dump(clf, 'trained_model.pkl')
    joblib.dump(X_train.columns, 'model_columns.pkl')
    joblib.dump(clf.classes_, 'model_classes.pkl')  # Save the class labels
    joblib.dump(scaler, 'scaler.pkl')  # Save the fitted scaler
    return clf

# Function to load the model and preprocessing parameters
def load_model():
    try:
        model = joblib.load('trained_model.pkl')
        model_columns = joblib.load('model_columns.pkl')
        model_classes = joblib.load('model_classes.pkl')
        scaler = joblib.load('scaler.pkl')
    except FileNotFoundError:
        model = train_model()
        model_columns = joblib.load('model_columns.pkl')
        model_classes = joblib.load('model_classes.pkl')
        scaler = joblib.load('scaler.pkl')
    return model, model_columns, model_classes, scaler

# Streamlit app
st.title('Obesity Disease Risk Prediction')

st.sidebar.header('User Input Features')

def user_input_features():
    Gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    Age = st.sidebar.slider('Age', 0, 100, 25)
    Height = st.sidebar.slider('Height (cm)', 100, 250, 170)
    Weight = st.sidebar.slider('Weight (kg)', 30, 200, 70)
    family_history_with_overweight = st.sidebar.selectbox('Family history with overweight', ['yes', 'no'])
    FAVC = st.sidebar.selectbox('FAVC (Frequent consumption of high caloric food)', ['yes', 'no'])
    FCVC = st.sidebar.slider('FCVC (Frequency of consumption of vegetables)', 1, 3, 2)
    NCP = st.sidebar.slider('NCP (Number of main meals)', 1, 4, 3)
    CAEC = st.sidebar.selectbox('CAEC (Consumption of food between meals)', ['no', 'Sometimes', 'Frequently', 'Always'])
    SMOKE = st.sidebar.selectbox('SMOKE', ['yes', 'no'])
    CH2O = st.sidebar.slider('CH2O (Consumption of water daily)', 1, 3, 2)
    SCC = st.sidebar.selectbox('SCC (Calories consumption monitoring)', ['yes', 'no'])
    FAF = st.sidebar.slider('FAF (Physical activity frequency)', 0, 2, 1)
    TUE = st.sidebar.slider('TUE (Time using technology devices)', 0, 2, 1)
    CALC = st.sidebar.selectbox('CALC (Consumption of alcohol)', ['no', 'Sometimes', 'Frequently'])
    MTRANS = st.sidebar.selectbox('MTRANS (Transportation used)', ['Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking'])

    data = {
        'Gender': Gender,
        'Age': Age,
        'Height': Height,
        'Weight': Weight,
        'family_history_with_overweight': family_history_with_overweight,
        'FAVC': FAVC,
        'FCVC': FCVC,
        'NCP': NCP,
        'CAEC': CAEC,
        'SMOKE': SMOKE,
        'CH2O': CH2O,
        'SCC': SCC,
        'FAF': FAF,
        'TUE': TUE,
        'CALC': CALC,
        'MTRANS': MTRANS
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

if st.button('Predict'):
    # Load the model and preprocessing parameters
    model, model_columns, model_classes, scaler = load_model()

    # Define numerical and categorical features again for preprocessing
    numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

    # Manually preprocess the user input
    df[numerical_features] = scaler.transform(df[numerical_features])
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    # Ensure the user input has the same columns as the training data
    missing_cols = set(model_columns) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    df = df[model_columns]

    st.write("Preprocessed user input:")
    st.write(df)

    # Predict probabilities
    prediction_proba = model.predict_proba(df)[0]
    prediction_df = pd.DataFrame({
        'Class': model_classes,
        'Probability': prediction_proba
    }).sort_values(by='Probability', ascending=False)

    st.subheader('Prediction Probabilities')
    st.write(prediction_df)
