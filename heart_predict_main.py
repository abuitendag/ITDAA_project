import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import sqlite3
import matplotlib.pyplot as plt
from joblib import Parallel, delayed 
import joblib 

# Create Streamlit app
#Heading
st.title('Heart Disease Prediction')
st.write('This application makes use of a machine learning model to predict the likelihood of heart disease.')
st.write("Please provide your patient's details to make a prediction.")
st.write("The prediction will be provided automatically.")
    
# column for user input parameters
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 600px !important; 
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Patient Details:')

st.sidebar.markdown("""---""")

help_text = "Please provide accurate input for best results."
age = st.sidebar.slider('Age:', min_value=20, max_value=100, value=0)
st.sidebar.markdown("""---""")
sex = st.sidebar.radio('Sex:', ['Male', 'Female'])
sex = 1 if sex == 'Male' else 0
st.sidebar.markdown("""---""")
cp = st.sidebar.selectbox('Chest Pain Type:', ['Typical Angina', 'Atypical Angina', 'Non-anginal pain', 'Asymptomatic'], help=help_text)
if cp == 'Typical Angina':
    cp = 1
elif cp == 'Atypical Angina':
    cp = 2
elif cp == 'Non-anginal pain':
    cp= 3
else:
    cp = 4
    
st.sidebar.markdown("""---""")
trestbps = st.sidebar.slider(
    'Resting Blood Pressure (mm Hg):', 
    min_value=80, max_value=200, 
    value=0, help=help_text
    )

st.sidebar.markdown("""---""")
chol = st.sidebar.slider(
    'Serum Cholesterol (mg/dl):', min_value=100, max_value=600, value=0, help=help_text
    )
st.sidebar.markdown("""---""")
fbs = st.sidebar.radio(
    'Fasting Blood Sugar above 120 mg/dl:', ['No', 'Yes'], help=help_text
    )
fbs = 1 if fbs == 'Yes' else 0  # Convert to integer
st.sidebar.markdown("""---""")
restecg = st.sidebar.selectbox(
    'Resting Electrocardiographic Results:', ['Normal', "Abnormal", "Ventricular Hypertrophy"], help=help_text
    )
if restecg == 'Normal':
    restecg = 0
elif restecg == 'Abnormal':
    restecg = 1
else: restecg = 2
#Add different values
st.sidebar.markdown("""---""")
thalach = st.sidebar.slider(
    'Maximum Heart Rate Achieved:', min_value=60, max_value=220, value=0, help=help_text
    )
st.sidebar.markdown("""---""")
exang = st.sidebar.radio(
    'Exercise Induced Angina:', ['No', 'Yes']
    )
exang = 1 if exang == 'Yes' else 0  # Convert to integer
st.sidebar.markdown("""---""")
oldpeak = st.sidebar.slider(
    'ST Depression Induced by Exercise Relative to Rest:', min_value=0.0, max_value=6.2, value=0.0, help=help_text
    )
st.sidebar.markdown("""---""")
slope = st.sidebar.selectbox(
    'Slope of the Peak Exercise ST Segment:', ['Upsloping', 'Flat', 'Downsloping'], help=help_text
    )
if slope == 'Upsloping':
    slope = 1
elif slope == 'Flat':
    slope = 2
else:
    slope = 3
st.sidebar.markdown("""---""")
ca = st.sidebar.selectbox(
    'Number of Major Vessels Colored by Flourosopy:', ['Mild','Moderate','Severe'], help=help_text
    )
if ca == 'Mild':
    ca = 0
elif ca == 'Moderate':
    ca = 1
else:
    ca = 3
st.sidebar.markdown("""---""")
thal = st.sidebar.selectbox(
    'Thalassemia:', ['Normal', 'Fixed Defect', 'Reversible Defect'], help=help_text
    )
if thal:
    thal = 1
elif thal:
    thal = 2
else:
    thal = 3
    
col1,col2 = st.columns(2)   
# Display prediction
with col1:
    st.subheader('Prediction:',)   
    

def show_result():
    model_run = joblib.load('final/svm_model.pkl')
    
    user_input = {
        "age": [age],
        "sex": [sex],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs],
        "restecg": [restecg],
        "thalach": [thalach],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "slope": [slope],
        "ca": [ca],
        "thal": [thal]
    }

    #turn user inputs into dataframe
    user_df = pd.DataFrame(user_input)
    
    #make prediction on user inputs
    prediction = model_run.predict(user_df)
    
    if prediction[0] == 1:
        with col2:
            st.error('Heart Disease Predicted')
    else:
        with col2:
            st.success('No Heart Disease Predicted')
         
#display results once button is pressed
if st.button(label="Calculate",help="Run the prediction", ):
    show_result()

    