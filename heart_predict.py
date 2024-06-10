import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sqlite3

df = pd.read_csv('heart.csv')

conn = sqlite3.connect('ml_data_.db')
query = "SELECT * FROM Heart"
df = pd.read_sql_query(query, conn)

predict = LogisticRegression()

# Create Streamlit app
#Heading
st.title('Heart Disease Prediction')
st.write('This application makes use of a machine learning model to predict the likelihood of heart disease.')
st.write("Please provide your patient's details to make a prediction.")
st.write("The prediction will be provided automatically.")
    
# Sidebar for user input parameters
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

st.sidebar.title('Patient Details')

st.sidebar.markdown("""---""")

age = st.sidebar.slider('Age:', min_value=20, max_value=100, value=60)
st.sidebar.markdown("""---""")
sex = st.sidebar.radio('Sex:', ['Male', 'Female'])
sex = 1 if sex == 'Male' else 0
st.sidebar.markdown("""---""")
cp = st.sidebar.selectbox('Chest Pain Type:', ['Typical Angina', 'Atypical Angina', 'Non-anginal pain', 'Asymptomatic'])
if cp == 'Typical Angina':
    cp = 1
elif cp == 'Atypical Angina':
    cp = 2
elif cp == 'Non-anginal pain':
    cp= 3
else:
    cp = 4
    

st.sidebar.markdown("""---""")
trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg):', min_value=80, max_value=200, value=120)
st.sidebar.markdown("""---""")
chol = st.sidebar.slider('Serum Cholesterol (mg/dl):', min_value=100, max_value=600, value=200)
st.sidebar.markdown("""---""")
fbs = st.sidebar.radio('Fasting Blood Sugar above 120 mg/dl:', ['No', 'Yes'])
fbs = 1 if fbs == 'Yes' else 0  # Convert to integer
st.sidebar.markdown("""---""")
restecg = st.sidebar.selectbox('Resting Electrocardiographic Results:', ['Normal', "Abnormal", "Ventricular Hypertrophy"])
if restecg == 'Normal':
    restecg = 0
elif restecg == 'Abnormal':
    restecg = 1
else: restecg = 2
#Add different values
st.sidebar.markdown("""---""")
thalach = st.sidebar.slider('Maximum Heart Rate Achieved:', min_value=60, max_value=220, value=150)
st.sidebar.markdown("""---""")
exang = st.sidebar.radio('Exercise Induced Angina:', ['No', 'Yes'])
exang = 1 if exang == 'Yes' else 0  # Convert to integer
st.sidebar.markdown("""---""")
oldpeak = st.sidebar.slider('ST Depression Induced by Exercise Relative to Rest:', min_value=0.0, max_value=6.2, value=1.0)
st.sidebar.markdown("""---""")
slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment:', ['Upsloping', 'Flat', 'Downsloping'])
if slope == 'Upsloping':
    slope = 1
elif slope == 'Flat':
    slope = 2
else:
    slope = 3
st.sidebar.markdown("""---""")
ca = st.sidebar.selectbox('Number of Major Vessels Colored by Flourosopy:', ['Mild','Moderate','Severe'])
if ca == 'Mild':
    ca = 0
elif ca == 'Moderate':
    ca = 1
else:
    ca = 3
st.sidebar.markdown("""---""")
thal = st.sidebar.selectbox('Thalassemia:', ['Normal', 'Fixed Defect', 'Reversible Defect'])
if thal:
    thal = 1
elif thal:
    thal = 2
else:
    thal = 3
st.sidebar.button(label="Process",)

    
# Separate features and target variable
X = df.drop(columns=["target"])  # Features
y = df["target"] 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
# Define categorical and numerical features
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(include=["int", "float"]).columns.tolist()

# Create preprocessing pipelines for both types of features
categorical_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

numerical_pipeline = make_pipeline(
    SimpleImputer(strategy="mean"),
    StandardScaler()
)

# Combine both pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)
#Logistic Regression is the model used
# Create pipelines
logistic_pipeline = make_pipeline(preprocessor, LogisticRegression())

# Train models
logistic_pipeline.fit(X_train, y_train)

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

user_df = pd.DataFrame(user_input)

# Make prediction
prediction = logistic_pipeline.predict(user_df)

# Display prediction
st.subheader('Prediction')
result_placeholder = st.empty()

if 'prediction' in locals():
    result_placeholder.write('Heart Disease' if prediction[0] == 1 else 'No Heart Disease')
    