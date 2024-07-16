from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)


rfmodel = pickle.load(open('rfmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))


input_columns = [
    'Age', 'BusinessTravel', 'Department', 'DistanceFromHome', 'Education', 
    'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime', 'DailyRate', 
    'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel', 
    'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 
    'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 
    'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 
    'YearsWithCurrManager'
]


encoded_columns = [
    'Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome',
    'MonthlyRate', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
    'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
    'Department_Research & Development', 'Department_Sales',
    'EducationField_Life Sciences', 'EducationField_Marketing',
    'EducationField_Medical', 'EducationField_Other', 'EducationField_Technical Degree',
    'Gender_Male', 'JobRole_Human Resources', 'JobRole_Laboratory Technician',
    'JobRole_Manager', 'JobRole_Manufacturing Director', 'JobRole_Research Director',
    'JobRole_Research Scientist', 'JobRole_Sales Executive', 'JobRole_Sales Representative',
    'MaritalStatus_Married', 'MaritalStatus_Single', 'OverTime_Yes'
]


numerical_columns = [
    'Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome', 
    'MonthlyRate', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']  
        
        
        df = pd.DataFrame([data])
        
        
        numerical_df = df[numerical_columns]
        categorical_df = df.drop(columns=numerical_columns)
        
        
        scaled_numerical_df = pd.DataFrame(scaler.transform(numerical_df), columns=numerical_columns)
        
        
        final_df = pd.concat([scaled_numerical_df, categorical_df], axis=1)
        
        
        final_df = final_df[encoded_columns]
        
        
        prediction = rfmodel.predict(final_df)
        
        
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        data = {col: request.form.get(col) for col in input_columns}
        
        
        df = pd.DataFrame([data])
        
        
        numerical_df = df[numerical_columns]
        categorical_df = df.drop(columns=numerical_columns)
        
        
        scaled_numerical_df = pd.DataFrame(scaler.transform(numerical_df), columns=numerical_columns)
        
        
        final_df = pd.concat([scaled_numerical_df, categorical_df], axis=1)
        
        
        final_df = final_df[encoded_columns]
        
        
        prediction = rfmodel.predict(final_df)[0]
        
        
        return render_template("home.html", predicted_text="Is there risk of Attrition: {}".format(prediction))
    
    except Exception as e:
        return render_template("home.html", predicted_text="Error: {}".format(str(e)))

if __name__ == '__main__':
    app.run(debug=True)
