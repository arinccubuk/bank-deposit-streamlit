import streamlit as st
import pandas as pd
import joblib

#load model
model = joblib.load("logistic_regression_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("Bank Term Deposit Prediction")

st.write("Enter client information to predict whether the client will subscribe to a bank term deposit.")

#input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
duration = st.number_input("Last contact duration (seconds)", min_value=0, value=100)
campaign = st.number_input("Number of contacts during this campaign", min_value=1, value=1)

job = st.selectbox("Job", [
    "admin", "blue-collar", "technician", "services",
    "management", "retired", "entrepreneur", "self-employed",
    "housemaid", "student", "unemployed", "unknown"
])

marital = st.selectbox("Marital status", ["single", "married", "divorced", "unknown"])
education = st.selectbox("Education", [
    "basic.4y", "basic.6y", "basic.9y", 
    "high.school", "professional.course", 
    "university.degree", "unknown"
])
preprocessor = model.named_steps['preprocessor']

categorical_cols = set()
for name, transformer, cols in preprocessor.transformers_:
  # columntransformer stores column lists for each transformer
  if name == "cat":
    if isinstance(cols, (list, tuple, pd.Index)):
      categorical_cols.update(list(cols))
    else:
      # in case cols is a selector fallback to empty
      pass

# build a full input row with safe defaults
row ={}
for col in feature_names:
  if col in categorical_cols:
    row[col] = "unknown" #default categorical value
  else:
    row[col] = 0 #default numeric value
    
#user-provided values
if "age" in row: row["age"] = int(age)
if "duration" in row: row["duration"] = int(duration)
if "campaign" in row: row["campaign"] = int(campaign)
if "job" in row: row["job"] = job
if "marital" in row: row["marital"] = marital
if "education" in row: row["education"] = education

#create dataframe
input_df = pd.DataFrame([row], columns=feature_names)

#prediction
if st.button("Predict"):
  try:
    pred = model.predict(input_df)[0]
    if pred == "yes":
      st.success("The client is likely to subscribe to a bank term deposit.")
    else:
      st.error("The client is not likely to subscribe to a bank term deposit.")
  except Exception as e:
    st.error("There is an error during prediction")
    st.exception(e)
