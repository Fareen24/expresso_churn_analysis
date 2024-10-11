# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib


# Function to load the saved Random Forest model and scaler
def load_randomforest():
    model = joblib.load('random_forest_model.pkl') 
    scaler = joblib.load('scaler_balanced.pkl')  
    return model, scaler

# Function for model selection buttons
def selected_model():
    
    col1 = st.columns(1)[0]  

    with col1:
        RF = st.button('Random Forest')
        if RF:
            st.session_state['selected_model'] = 'random_forest'
            st.session_state['model'], st.session_state['scaler'] = load_randomforest()  # Load Random Forest model
    
    
# Function to make predictions using the selected model
def make_prediction(features: pd.DataFrame):
    if 'model' not in st.session_state or 'scaler' not in st.session_state:
        st.error("Please select a model first!")
        return None, None
    
    # Scale the features using the saved scaler
    scaled_features = st.session_state['scaler'].transform(features)

    # Make prediction using the selected model
    model = st.session_state['model']
    prediction = model.predict(scaled_features)
    probability = model.predict_proba(scaled_features)
    
    return prediction, probability

# Streamlit app interface
st.title("Churn Prediction App")

# Call the model selection function
selected_model()

# Input form for features (example with placeholder feature names)
st.subheader("Enter Client Information")
montant = st.number_input("Montant", min_value=0.0, step=0.1, key="montant")
frequence_rech = st.number_input("Frequence Recharge", min_value=0, step=1, key="frequence_rech")
revenue = st.number_input("Revenue", min_value=0.0, step=0.1, key="revenue")
arpu_segment = st.number_input("ARPU Segment", min_value=0.0, step=0.1, key="arpu_segment")
frequence = st.number_input("Frequence", min_value=0, step=1, key="frequence")
data_volume = st.number_input("Data Volume", min_value=0.0, step=0.1, key="data_volume")
on_net = st.number_input("On Net", min_value=0, max_value=1000, step=1, key="on_net")
orange = st.number_input("Orange", min_value=0, max_value=1000, step=1, key="orange")
tigo = st.number_input("Tigo", min_value=0, max_value=1000, step=1, key="tigo")
zone1 = st.number_input("Zone1", min_value=0, max_value=1, step=1, key="zone1")
zone2 = st.number_input("Zone2", min_value=0, max_value=2, step=1, key="zone2")
mrg = st.number_input("MRG", min_value=0, max_value=10000, step=1, key="mrg")
regularity = st.number_input("Regularity", min_value=0, max_value=100, step=1, key="regularity")
freq_top_pack = st.number_input("Freq Top Pack", min_value=0, max_value=100, step=1, key="freq_top_pack")
tenure = st.number_input("Tenure", min_value=0, max_value=100, step=1, key="tenure")
top_pack = st.number_input("Top Pack", min_value=0, max_value=10000, step=1, key="top_pack")

# Create a DataFrame from the input
input_features = pd.DataFrame({
    'montant': [montant],
    'frequence_rech': [frequence_rech],
    'revenue': [revenue],
    'arpu_segment': [arpu_segment],
    'frequence': [frequence],
    'data_volume': [data_volume],
    'on_net': [on_net],
    'orange': [orange], 
    'tigo': [tigo],
    'zone1': [zone1],
    'zone2': [zone2], 
    'mrg': [mrg], 
    'regularity': [regularity], 
    'freq_top_pack': [freq_top_pack], 
    'tenure':[tenure],
    'top_pack': [top_pack]
})

# Prediction button
if st.button("Predict Churn"):
    prediction, probability = make_prediction(input_features)
    
    if prediction is not None:
        st.write(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
        st.write(f"Probability of Churn: {probability[0][1]:.2f}")
