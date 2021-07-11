import pandas as pd
import seaborn as sns
X=pd.read_csv("diabeties.csv")
X.head()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
X_train, X_test, y_train, y_test = train_test_split(X[['Pregnancies', 'Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']], X['Outcome'], test_size=0.3, random_state=109)
logisticRegr = LogisticRegression(C=1)
logisticRegr.fit(X_train, y_train)
y_pred = logisticRegr.predict(X_test)
pickle_out = open("logisticRegr.pkl", "wb") 
pickle.dump(logisticRegr, pickle_out) 
pickle_out.close()
import numpy as np
import streamlit as st
import pickle
pickle_in = open('logisticRegr.pkl', 'rb')
classifier = pickle.load(pickle_in)
st.sidebar.header('Diabetes Prediction')
st.title('Diabetes Prediction(Only for females above 21years of    Age)')
name = st.text_input("Name:")
Pregnancies = st.number_input("No. of times pregnant:")
glucose = st.number_input("Plasma Glucose Concentration :")
bp =  st.number_input("Diastolic blood pressure (mm Hg):")
skin = st.number_input("Triceps skin fold thickness (mm):")
insulin = st.number_input("2-Hour serum insulin (mu U/ml):")
bmi = st.number_input("Body mass index (weight in kg/(height in m)^2):")
dpf = st.number_input("Diabetes Pedigree Function:")
age = st.number_input("Age:")
submit = st.button('Predict')
if submit:
        prediction = classifier.predict([[Pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        if prediction == 0:
            st.write('Congratulation',name,'You are not diabetic')
        else:
            st.write(name," we are really sorry to say but it seems like you are Diabetic.")