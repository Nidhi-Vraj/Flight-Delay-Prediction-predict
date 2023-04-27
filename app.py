#Importing the packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
import os
import pickle
from flask import Flask, render_template, request
import xgboost as xgb

#Loading the data
def load_data():
    df1 = pd.read_csv(r'data/ORD_airport_data.csv')
    df2 = pd.read_csv(r'data/DEN_airport_data.csv')
    df3 = pd.read_csv(r'data/ATL_airport_data.csv')
    df = pd.concat([df1,df2,df3])
    status = []
    for value in zip(df['DepDel15'],df['ArrDel15']):
        if value[0] == 0 or value[1] == 0:
            status.append(0)
        else:
            status.append(1)
    df['FLIGHT_STATUS'] = status
    df = df[['Origin','Dest','Year','Month','DayofMonth','Operating_Airline ','DepTime','ArrTime','FLIGHT_STATUS']]
    return df

#Preprocessing the data
def preprocessing(df):
    df = df.dropna()
    le = LabelEncoder()
    le2 = LabelEncoder()
    df['Origin'] = le.fit_transform(df['Origin'])
    df['Dest'] = le2.fit_transform(df['Dest'])
    df['Operating_Airline '] = le.fit_transform(df['Operating_Airline '])
    scaler = StandardScaler()
    df[['DepTime','ArrTime']] = scaler.fit_transform(df[['DepTime','ArrTime']])
    with open('scaler.pickle', 'wb') as f:
        pickle.dump(scaler, f)
    with open('le.pickle', 'wb') as f:
        pickle.dump(le2, f)
    return df

#Model Creation
def model_create(df):
    X = df.drop('FLIGHT_STATUS', axis=1)
    y = df['FLIGHT_STATUS']
    model = xgb.XGBClassifier()
    model.fit(X, y)
    #Saving the model
    filename = 'model.pickle'
    with open(filename,'wb') as f:
        pickle.dump(model,f)
    return 'Model Creation successful!'

#Model prediction
def predict_ans(new_flight):
    filename = 'model.pickle'
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    with open('le.pickle', 'rb') as f:
        le2 = pickle.load(f)
    with open('scaler.pickle', 'rb') as f:
        scaler = pickle.load(f)
    new_flight['Dest'] = le2.transform(new_flight['Dest'])
    new_flight[['DepTime','ArrTime']] = scaler.transform(new_flight[['DepTime','ArrTime']])
    percentage = round(model.predict_proba(new_flight) [:,1][0] * 100,2)
    answer = model.predict(new_flight)
    return percentage,answer

#Flask app code
app = Flask(__name__,template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def generate():
    origin = request.form['origin']
    origin = int(origin)
    origin_dict = {0:'ATL',1:'DEN',2:'ORD'}
    destination = request.form['destination']
    if origin_dict[origin] == destination:
            return render_template('index.html',error_message =  "Invalid input. Please try again.")
    date = request.form['date_input']
    airline_name = request.form['airline-name']
    airline_name = int(airline_name)
    arrival_time = request.form['arrival-time']
    departure_time = request.form['departure-time']
    year,month,day = date.split('-')
    arrival_time = arrival_time.replace(':', '') + '.00'
    departure_time = departure_time.replace(':', '') + '.00'
    new_flight = pd.DataFrame({'Origin':[origin],'Dest': [destination],'Year': [int(year)],'Month':[int(month)],'DayofMonth':[int(day)],'Operating_Airline ': [airline_name],'DepTime': [float(departure_time)],'ArrTime':[float(arrival_time)]})
    if os.path.isfile('model.pickle'):
        percentage,answer = predict_ans(new_flight)
    else:
        df = load_data()
        df = preprocessing(df)
        model_create(df)
        percentage,answer = predict_ans(new_flight)
    return render_template('result.html', answer = answer, percentage=percentage)
    
if __name__ == '__main__':
    app.run(debug=True)