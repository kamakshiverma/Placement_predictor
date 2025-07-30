# Smart Student Placement Predictor

This project is a machine learning-based web application that predicts whether a student will get placed based on their academic performance, skills, and experience. It uses two machine learning models—Logistic Regression and Random Forest Classifier—to provide predictions and is deployed using a simple Streamlit web interface.

## Project Overview

The goal of this project is to help academic institutions or training and placement cells predict student placement outcomes early, based on various measurable attributes. The system uses historical student data to learn patterns and then predicts whether a new student is likely to be placed or not.

## How It Works

1. **Data Source**:
   - Student data is fetched from a MySQL database using `mysql-connector-python`.
   - Data includes features like CGPA, IQ, past semester results, project count, communication skills, etc.

2. **Data Preprocessing**:
   - Outliers are removed using the IQR method.
   - Categorical variables such as `Internship_Experience` and `Placement` are mapped to numeric values.
   - Unnecessary columns like `College_ID` are dropped.

3. **Model Training**:
   - Two models are trained:
     - Logistic Regression (with `class_weight='balanced'`)
     - Random Forest Classifier
   - The dataset is split into training and test sets using `train_test_split`.
   - Model performance is evaluated using accuracy, recall, precision, and F1-score.
   - Cross-validation (5-fold) is used to ensure generalizability.

4. **Prediction**:
   - Users can input new student data through a Streamlit form.
   - The trained model predicts whether the student will be placed or not.
   - The model is saved using `joblib` and loaded into the Streamlit app for prediction.

## Input Features

- IQ
- Previous Semester Result
- CGPA
- Academic Performance
- Internship Experience (0 = No, 1 = Yes)
- Extra-Curricular Score
- Communication Skills
- Number of Projects Completed

## Output

The model returns:
- `1` for "Placed"
- `0` for "Not Placed"

The Streamlit UI displays a corresponding message to the user.

## How to Run the Application

### Prerequisites

- Python 3.8+
- MySQL Server with student data
- VS Code or any IDE
- pip (Python package manager)

### Steps

1. Clone the repository:

```bash
git clone https://github.com/your-username/student-placement-predictor.git
cd student-placement-predictor
