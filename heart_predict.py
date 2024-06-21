import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import sqlite3
import matplotlib.pyplot as plt

#connect to db
conn = sqlite3.connect('itdaa.db')
query = "SELECT * FROM heart"
df = pd.read_sql_query(query, conn, index_col=None)

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

st.sidebar.title('Patient Details')

st.sidebar.markdown("""---""")

age = st.sidebar.slider('Age:', min_value=20, max_value=100, value=0)
st.sidebar.markdown("""---""")
sex = st.sidebar.radio('Sex:', ['Male', 'Female'])
sex = 1 if sex == 'Male' else 0
st.sidebar.markdown("""---""")
cp = st.sidebar.selectbox('Chest Pain Type:', ['Typical Angina', 'Atypical Angina', 'Non-anginal pain', 'Asymptomatic'], help="This is to help")
if cp == 'Typical Angina':
    cp = 1
elif cp == 'Atypical Angina':
    cp = 2
elif cp == 'Non-anginal pain':
    cp= 3
else:
    cp = 4
    
st.sidebar.markdown("""---""")
trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg):', min_value=80, max_value=200, value=0, help="This is to help")
st.sidebar.markdown("""---""")
chol = st.sidebar.slider('Serum Cholesterol (mg/dl):', min_value=100, max_value=600, value=0, help="This is to help")
st.sidebar.markdown("""---""")
fbs = st.sidebar.radio('Fasting Blood Sugar above 120 mg/dl:', ['No', 'Yes'], help="This is to help")
fbs = 1 if fbs == 'Yes' else 0  # Convert to integer
st.sidebar.markdown("""---""")
restecg = st.sidebar.selectbox('Resting Electrocardiographic Results:', ['Normal', "Abnormal", "Ventricular Hypertrophy"], help="This is to help")
if restecg == 'Normal':
    restecg = 0
elif restecg == 'Abnormal':
    restecg = 1
else: restecg = 2
#Add different values
st.sidebar.markdown("""---""")
thalach = st.sidebar.slider('Maximum Heart Rate Achieved:', min_value=60, max_value=220, value=0, help="This is to help")
st.sidebar.markdown("""---""")
exang = st.sidebar.radio('Exercise Induced Angina:', ['No', 'Yes'])
exang = 1 if exang == 'Yes' else 0  # Convert to integer
st.sidebar.markdown("""---""")
oldpeak = st.sidebar.slider('ST Depression Induced by Exercise Relative to Rest:', min_value=0.0, max_value=6.2, value=0.0, help="This is to help")
st.sidebar.markdown("""---""")
slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment:', ['Upsloping', 'Flat', 'Downsloping'], help="This is to help")
if slope == 'Upsloping':
    slope = 1
elif slope == 'Flat':
    slope = 2
else:
    slope = 3
st.sidebar.markdown("""---""")
ca = st.sidebar.selectbox('Number of Major Vessels Colored by Flourosopy:', ['Mild','Moderate','Severe'], help="This is to help")
if ca == 'Mild':
    ca = 0
elif ca == 'Moderate':
    ca = 1
else:
    ca = 3
st.sidebar.markdown("""---""")
thal = st.sidebar.selectbox('Thalassemia:', ['Normal', 'Fixed Defect', 'Reversible Defect'], help="This is to help")
if thal:
    thal = 1
elif thal:
    thal = 2
else:
    thal = 3
    
col1,col2 = st.columns(2)   
# Display prediction
with col1:
    st.subheader('Prediction',)    
    result_placeholder = st.empty()
    
# Separate features and target variable
X = df.drop(columns="target", axis=1)  # Features
y = df["target"] 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
# Define categorical and numerical features
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(include=["int", "float"]).columns.tolist()

# Create preprocessing pipelines for both types of features
categorical_pipeline = make_pipeline(
    OneHotEncoder(handle_unknown="ignore")
)

#Scaling done to numerical columns
numerical_pipeline = make_pipeline(
    StandardScaler()
)

# Combine both pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)
#SVM with polynominal kernel is used
svm_model = SVC(kernel='poly')
# Create pipelines
svm_pipeline = make_pipeline(preprocessor, SVC())

# Train models
svm_pipeline.fit(X_train, y_train)

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

# Make prediction on user inputs
prediction = svm_pipeline.predict(user_df)


graph_placeholder = st.empty()
plt.figure(figsize=(10, 6))
plt.figure(facecolor='grey')
graph_c = plt.axes()
graph_c.set_facecolor("grey")
plt.scatter(X_train['age'], X_train['trestbps'],  cmap='coolwarm', alpha=0.7, label='Training Data')
plt.scatter(user_df['age'], user_df['trestbps'], color='red', marker='x', s=100, label='User Input')
plt.xlabel('Age')
plt.ylabel('Resting Blood Pressure (mm Hg)')
plt.title('Heart Disease Prediction - User Input vs Training Data')
plt.legend()
plt.grid(True)


with col2:    
    # Convert Matplotlib plot to Plotly for Streamlit compatibility
    st.pyplot(plt)



if 'prediction' in locals():
    result_placeholder.write('Heart Disease' if prediction[0] == 1 else 'No Heart Disease')
    