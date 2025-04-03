# Loan Approval Prediction

https://loan-status-prediction-gewycbcsdqgacavp98vsmc.streamlit.app/

## Overview
This project is a machine learning-based loan approval prediction system built using Python and Streamlit. It takes user inputs such as income, credit history, and property details to predict whether a loan application will be approved or not.

## Features
- Data preprocessing (handling missing values, encoding categorical variables, and standardization)
- Random Forest classifier for loan status prediction
- Streamlit-based interactive UI for easy user input and real-time predictions

## Requirements
Ensure you have the following dependencies installed:

```sh
pip install streamlit pandas numpy scikit-learn
```

## How to Run
1. Clone the repository or download the script.
2. Place the dataset (`train_u6lujuX_CVtuZ9i (1).csv`) in the same directory as the script.
3. Run the following command in the terminal:

```sh
streamlit run script.py
```

Replace `script.py` with the actual filename of your script.

## Usage
1. Open the Streamlit UI in your browser.
2. Enter the required loan application details.
3. Click on "Predict Loan Status" to get the prediction result.

## Dataset
The dataset contains loan application details such as:
- Gender
- Marital Status
- Dependents
- Education
- Self Employment
- Income details
- Credit History
- Loan Amount and Term
- Property Area

## Model
The model used for prediction is a **Random Forest Classifier** trained on the preprocessed dataset. The data is split into training and testing sets, and numerical features are standardized to improve performance.

## License
This project is for educational purposes and does not guarantee real-world loan approvals.

