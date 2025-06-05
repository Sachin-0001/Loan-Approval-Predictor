import joblib
import streamlit as st
import numpy as np
import pandas as pd

# Load ML artifacts
model = joblib.load('classifier.pkl')
label_encoder = joblib.load('label_encoder.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# Page configuration
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .header {
        font-size: 36px !important;
        color: #34eb3a;
        text-align: center;
        margin-bottom: 30px;
    }
    .stNumberInput, .stSelectbox {
        margin-bottom: 15px;
    }
    .approved-text {
        color: #34eb3a;
        font-size: 32px;
        text-align: center;
    }
    .rejected-text {
        color: #ff0000;
        font-size: 32px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def user_input_features():
    # App Header
    st.markdown('<p class="header">🏦 Smart Loan Approval Predictor</p>', unsafe_allow_html=True)
    
    # Using columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Details")
        Gender = st.selectbox('Gender', ['Male', 'Female'], help="Select applicant's gender")
        Married = st.selectbox('Marital Status', ['Yes', 'No'], help="Marital status of applicant")
        Dependents = st.selectbox('Number of Dependents', ['0', '1', '2', '3'], 
                                help="Number of people dependent on applicant")
        Education = st.selectbox('Education Level', ['Graduate', 'Not Graduate'], 
                               help="Highest education qualification")
        
    with col2:
        st.subheader("Financial Details")
        Credit_History = st.selectbox('Credit History', [1.0, 0.0], 
                                    help="1.0 = Good credit history\n0.0 = Poor credit history")
        Loan_Amount_Term = st.number_input('Loan Term (months)', min_value=0, 
                                         help="Duration of loan in months")
        ApplicantIncome = st.number_input('Annual Applicant Income (₹)', min_value=0, 
                                        help="Yearly income before taxes")/12
        CoapplicantIncome = st.number_input('Annual Co-applicant Income (₹)', min_value=0, 
                                          help="Yearly income of co-applicant")/12
        LoanAmount = st.number_input('Loan Amount (₹)', min_value=0, 
                                   help="Total loan amount requested")/1000

    # Feature engineering
    loanAmount_log = np.log(LoanAmount) if LoanAmount > 0 else 0
    TotalIncome = ApplicantIncome + CoapplicantIncome

    data = {
        'Gender': Gender,
        'Married': Married,
        'Dependents': Dependents,
        'Education': Education,
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': Credit_History,
        'loanAmount_log': loanAmount_log,
        'TotalIncome': TotalIncome
    }
    return pd.DataFrame([data])

# Main app logic
input_df = user_input_features()

# Prediction section
st.write("")  # Add spacing
predict_col, _ = st.columns([1, 2])  # Centered prediction button
with predict_col:
    if st.button('🔍 Predict Loan Approval', use_container_width=True):
        try:
            X_processed = preprocessor.transform(input_df)
            pred = model.predict(X_processed)
            result = label_encoder.inverse_transform(pred)[0]
            status = "Approved" if result == "Y" else "Rejected"
            
            # Styled prediction result
            if status == "Approved":
                st.balloons()
                st.success("Loan Approved! 🎉")
            else:
                st.error("Loan Rejected! ❌")
            
            # css_class = "approved-text" if status == "Approved" else "rejected-text"
            # st.markdown(
            #     f'<h2 class="{css_class}">Loan Status: {status}!</h2>', 
            #     unsafe_allow_html=True
            # )
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
# Footer
st.write("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🔒 Your data is processed securely and never stored</p>
    <p>Sachin Suresh© | All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
