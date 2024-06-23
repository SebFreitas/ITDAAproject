#Sebastian Freitas T64M7SQJ3 Q1

import sqlite3
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

heartdb_csv = 'heart.csv'
df = pd.read_csv(heartdb_csv, delimiter=';')

# Connect to the database and save the dataframe to the database
conn = sqlite3.connect('heart.db')
df.to_sql('Patients', conn, if_exists='replace', index=False)
query = "SELECT * FROM Patients"
df = pd.read_sql_query(query, conn)

conn.commit()
conn.close()

#Q2a
df = df.dropna()
df = df.drop_duplicates()

print(df)
print(df.isnull().sum())
print(df.duplicated())

#Mapping
'''
var_mapping = {
    'sex': {0: 'Female', 1: 'Male'},
    'target': {0: 'Healthy', 1: 'Heart Disease'},
    'cp': {0: 'typical angina', 1: 'atypical angina', 2: 'non-anginal pain', 3: 'asymptomatic'},
    'fbs':{0: 'false', 1: 'true'},
    'restecg':{0:'normal',1:'abnormal',2:'ventricular hypertrohpy'},
    'exang':{0:'no',1:'yes'},
    'slope':{0:'upsloping',1:'flat',2:'downslopping'},
    'thal':{0:'unknown', 1:'normal', 2:'fixed defect', 3:'reversible defect'},
    }

for x, mapping in var_mapping.items():
    df[x] = df[x].map(mapping)
'''
    
# Categorical Variables
categorical_vars = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Numeric Variables
numerical_vars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Target Variable
target_var = 'target'

#Q2b

for c_var in categorical_vars:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=c_var, hue=target_var, data=df)
    plt.title('Distribution of Classes for '+ c_var.capitalize() + ' based on Target Variable', fontweight='bold')
    plt.xlabel(c_var.capitalize(), fontweight='bold')
    plt.ylabel('Count', fontweight='bold')
    plt.legend(title='Target')
    plt.show()

#Q2c

for n_var in numerical_vars:
    plt.figure(figsize=(20, 10))
    sns.countplot(x=n_var, hue=target_var, data=df)
    plt.title('Distribution of Classes for '+ n_var.capitalize() + ' based on Target Variable', fontweight='bold')
    plt.xlabel(n_var.capitalize(), fontweight='bold')
    plt.ylabel('Count', fontweight='bold')
    plt.legend(title='Target')
    plt.xticks(rotation=90)
    plt.show()    
    
#Q3a
# Define features and target variable
y = df['target']
X = df.drop(columns=['target'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling for numerical variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machine': SVC(random_state=42)
}

# Train and evaluate each model
best_model = None
best_accuracy = 0

for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.2f}")
    
    # Save the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Save the best model to disk
joblib.dump(best_model, 'best_model.pkl')
print("Best model saved to 'best_model.pkl'")
