# %%

import pandas as pd
import mysql.connector

# %%
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='1234',   # change this
    database='placement_db'
)



# %%
# Read data into Pandas
df = pd.read_sql("SELECT * FROM students", conn)
#print(df.head())
print(df.describe())
Q1 = df['CGPA'].quantile(0.25)
Q3 = df['CGPA'].quantile(0.75)
IQR = Q3 - Q1

# %%
# Remove outliers
df = df[(df['CGPA'] >= Q1 - 1.5 * IQR) & (df['CGPA'] <= Q3 + 1.5 * IQR)]
# Convert to string first



# %%
print(df['Internship_Experience'])

# %%

df['Internship_Experience']= df['Internship_Experience'].map({'yes':1,'no':0})

print(df['Internship_Experience'])

# %%

df['Placement']= df['Placement'].map({'yes':1,'no':0})
print(df['Placement'])


# %%
df.drop(columns = 'College_ID',inplace = True)

# %%
print(df.head())
from sklearn.model_selection import train_test_split



# %%
from sklearn.model_selection import train_test_split

# %%
x = df.drop('Placement',axis = 1)

y = df['Placement']

# %%
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# %%
print(x_train)


# %%
print(y_test)

# %%
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# %%
print(x_train.dtypes)


# %%
print(x_train.isnull().sum())


# %%
model.fit(x_train, y_train)


# %%
y_pred = model.predict(x_test)

# %%
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# %%
model = LogisticRegression(class_weight='balanced',max_iter=1000)

# %%
model.fit(x_train,y_train)
{}

# %%
y_pred=model.predict(x_test)

# %%
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score,classification_report
print("Accuracy is: ", accuracy_score(y_pred,y_test))
print("recall score is:", classification_report(y_pred, y_test))

# %%
from sklearn.model_selection import cross_val_score

# %%
scores = cross_val_score(model,x,y,cv =5, scoring='recall')

# %%
print(scores)
print(scores.mean())

# %%
import joblib

joblib.dump(model, 'placement_model.pkl')




# %%
model = joblib.load('placement_model.pkl')

# %%
new_data = [[110, 7.5, 8.0, 0, 2, 0, 0, 6]]  # replace with your own

# Predict placement
prediction = model.predict(new_data)
print("Placed" if prediction[0] == 1 else "Not Placed")

# %%
importance = model.coef_[0]
for col, imp in zip(x_train.columns, importance):
    print(f"{col}: {imp:.3f}")


# %%
import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('placement_model.pkl')

# Page config
st.set_page_config(page_title="Smart Student Placement Predictor", layout="centered")

# Page title
st.markdown(
    "<h1 style='text-align: center; color: #4B8BBE;'>🎓 Smart Student Placement Predictor</h1>",
    unsafe_allow_html=True
)

# Instructions
st.markdown(
    """
    <div style='text-align: center; font-size: 16px; color: #6c757d;'>
    Enter student details below to check if they are likely to be placed.
    </div><br>
    """,
    unsafe_allow_html=True
)

# Input form
with st.form("placement_form"):
    col1, col2 = st.columns(2)

    with col1:
        iq = st.number_input("🧠 IQ Score", min_value=50.0, max_value=200.0, step=1.0)
        prev_sem = st.number_input("📊 Previous Semester Result (out of 10)", min_value=0.0, max_value=10.0, step=0.1)
        cgpa = st.number_input("🎯 CGPA", min_value=0.0, max_value=10.0, step=0.1)
        academic = st.number_input("📘 Academic Performance (Score)", min_value=0, max_value=10, step=1)

    with col2:
        internship = st.selectbox("💼 Internship Experience", options=["No", "Yes"])
        extra_curricular = st.number_input("🏅 Extra Curricular Score", min_value=0, max_value=10, step=1)
        communication = st.number_input("🗣️ Communication Skills", min_value=0, max_value=10, step=1)
        projects = st.number_input("📂 Number of Projects", min_value=0, max_value=10, step=1)

    submitted = st.form_submit_button("🔍 Predict Placement")

    if submitted:
        internship_val = 1 if internship == "Yes" else 0
        input_data = [[iq, prev_sem, cgpa, academic, internship_val, extra_curricular, communication, projects]]
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.success("🎉 The student is **likely to be placed!**")
        else:
            st.error("⚠️ The student is **not likely to be placed.**")


# %%
from sklearn.ensemble import RandomForestClassifier

# %%
rf_model = RandomForestClassifier(class_weight='balanced',n_estimators=100,random_state=42)

# %%
rf_model.fit(x_train, y_train)


# %%
rf_pred = rf_model.predict(x_test)

# %%
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print("accuracy is: ", accuracy_score(y_test,rf_pred))

# %%
print("classfication report is:", classification_report(y_test,rf_pred))

# %%
from sklearn.model_selection import cross_val_score
rf_scores = cross_val_score(model,x,y,cv =5,scoring = 'accuracy')

# %%
print(rf_scores)

# %%
import joblib

# %%
joblib.dump(rf_model,'rf_placement_model.pkl')

# %%
# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load models
log_model = joblib.load('placement_model.pkl')
rf_model = joblib.load('rf_placement_model.pkl')

# Set page config
st.set_page_config(page_title="Student Placement Predictor", layout="centered")

st.title("🎓 Smart Student Placement Predictor")
st.markdown("Predict if a student is likely to be placed based on academic and personal attributes.")

# Model selection
model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest"])

# Input fields
st.subheader("📥 Enter Student Details")

iq = st.number_input("IQ", min_value=50, max_value=200, step=1)
prev_result = st.number_input("Previous Semester Result", min_value=0.0, max_value=10.0, step=0.1)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1)
academic_perf = st.number_input("Academic Performance", min_value=0, max_value=10, step=1)
internship = st.selectbox("Internship Experience", ["No", "Yes"])
extra_curricular = st.number_input("Extra Curricular Score", min_value=0, max_value=10, step=1)
communication = st.number_input("Communication Skills", min_value=0, max_value=10, step=1)
projects = st.number_input("Number of Projects Completed", min_value=0, max_value=10, step=1)

# Convert categorical input
internship_val = 1 if internship == "Yes" else 0

input_data = np.array([[iq, prev_result, cgpa, academic_perf, internship_val, extra_curricular, communication, projects]])

# Predict button
if st.button("🎯 Predict Placement"):
    if model_choice == "Logistic Regression":
        model = log_model
    else:
        model = rf_model

    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.success(f"✅ Student is likely to be **Placed**! \n\n📊 Confidence: `{proba*100:.2f}%`")
    else:
        st.error(f"❌ Student is **Not Likely to be Placed**.\n\n📊 Confidence: `{(1-proba)*100:.2f}%`")

# Optional: Feature Importance for RF
if model_choice == "Random Forest":
    st.subheader("📌 Feature Importance (Random Forest)")
    importance = rf_model.feature_importances_
    features = ['IQ', 'Prev Result', 'CGPA', 'Academic', 'Internship', 'Extra Curricular', 'Communication', 'Projects']
    df_feat = pd.DataFrame({'Feature': features, 'Importance': importance})
    df_feat = df_feat.sort_values(by='Importance', ascending=False)

    st.bar_chart(df_feat.set_index("Feature"))


# %%



