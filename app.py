import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("train_u6lujuX_CVtuZ9i (1).csv")

# Drop Loan_ID since it's not useful for prediction
df = df.drop(columns=["Loan_ID"])

# Handling missing values
imputer = SimpleImputer(strategy="most_frequent")
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Encoding categorical variables
label_encoders = {}
categorical_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area", "Loan_Status"]

for col in categorical_cols:
    le = LabelEncoder()
    df_imputed[col] = le.fit_transform(df_imputed[col])
    label_encoders[col] = le

# Splitting features and target variable
X = df_imputed.drop(columns=["Loan_Status"])
y = df_imputed["Loan_Status"]

# Standardizing numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Training a model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

def predict_loan_status(input_data):
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data_as_numpy_array)
    prediction = model.predict(input_data_scaled)
    return "Loan Approved" if label_encoders['Loan_Status'].inverse_transform(prediction)[0] == 'Y' else "Loan Not Approved"

# Streamlit UI
st.title("Loan Approval Prediction")
st.write("Enter the details below to check loan approval status:")

# User input fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])
income = st.number_input("Applicant Income", min_value=0, step=1000)
co_income = st.slider("Coapplicant Income", min_value=0, max_value=50000, step=500)
loan_amount = st.slider("Loan Amount", min_value=0, max_value=500, step=10)
loan_term = st.selectbox("Loan Amount Term (in months)", [12, 24, 36, 48, 60, 120, 180, 240, 300, 360])
credit_history = st.selectbox("Credit History", [0, 1])
property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# Encoding user inputs
input_data = [
    label_encoders['Gender'].transform([gender])[0],
    label_encoders['Married'].transform([married])[0],
    label_encoders['Dependents'].transform([dependents])[0],
    label_encoders['Education'].transform([education])[0],
    label_encoders['Self_Employed'].transform([self_employed])[0],
    income, co_income, loan_amount, loan_term,
    credit_history, label_encoders['Property_Area'].transform([property_area])[0]
]

if st.button("Predict Loan Status"):
    result = predict_loan_status(input_data)
    st.write(result)
