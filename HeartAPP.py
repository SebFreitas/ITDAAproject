import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('./best_model.pkl')

# Define the Streamlit app
def main():
    st.title('Heart Disease Risk Prediction')

    # Input fields for doctor's input
    age = st.number_input('Age', min_value=0, max_value=120, value=50)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    cp = st.selectbox('Chest Pain Type', ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'])
    trestbps = st.number_input('Resting Blood Pressure', min_value=0, max_value=300, value=120)
    chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=0, max_value=1000, value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['true', 'false'])
    restecg = st.selectbox('Resting ECG Results', ['normal', 'abnormal', 'ventricular hypertrophy'])
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=250, value=150)
    exang = st.selectbox('Exercise Induced Angina', ['yes', 'no'])
    oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=9.0)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['upsloping', 'flat', 'downsloping'])
    ca = st.number_input('Number of Major Vessels Colored by Fluoroscopy', min_value=0, max_value=4, value=4)
    thal = st.selectbox('Thalassemia', ['unknown', 'normal', 'fixed defect', 'reversible defect'])

    # Create a DataFrame for prediction
    input_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    input_df = pd.DataFrame(input_data, index=[0])
    print(input_df)
    
    # Preprocess input data
    if sex == 'Male':
        input_df.loc[0, 'sex'] = 1
    elif sex == "Female":
        input_df.loc[0, 'sex'] = 0
    
    if cp == "typical angina":
        input_df.loc[0, 'cp'] = 0
    elif cp == "atypical angina":
        input_df.loc[0, 'cp'] = 1
    elif cp == "non-anginal pain":
        input_df.loc[0, 'cp'] = 2
    elif cp == "asymptomatic":
        input_df.loc[0, 'cp'] = 3
    
    if fbs == "true":
        input_df.loc[0, 'fbs'] = 1
    elif fbs == "false":
        input_df.loc[0, 'fbs'] = 0
        
    if restecg == "normal":
        input_df.loc[0, 'restecg'] = 0
    elif restecg == "abnormal":
        input_df.loc[0, 'restecg'] = 1
    elif restecg == "ventricular hypertrophy":
        input_df.loc[0, 'restecg'] = 2
        
    if exang == "yes":
        input_df.loc[0, 'exang'] = 1
    elif exang == "no":
        input_df.loc[0, 'exang'] = 0
        
    if slope == "upsloping":
        input_df.loc[0, 'slope'] = 0
    elif slope == "flat":    
        input_df.loc[0, 'slope'] = 1
    elif slope == "downsloping":
        input_df.loc[0, 'slope'] = 2
    
    if thal == "unknown":
        input_df.loc[0, 'thal'] = 0
    elif thal == "normal":
        input_df.loc[0, 'thal'] = 1
    elif thal == "fixed defect":
        input_df.loc[0, 'thal'] = 2
    elif thal == "reversible defect":
        input_df.loc[0, 'thal'] = 3

    print(input_df)
    input_df = input_df.to_numpy()
    
    # Predict and display result
    if st.button('Predict'):
        print(input_df)
        prediction = model.predict(input_df)
        print(prediction)
        if prediction == 1:
            st.write('The patient is likely to have heart disease.')
        else:
            st.write('The patient is likely healthy.')

if __name__ == '__main__':
    main()
