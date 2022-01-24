from django.shortcuts import render
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import datasets


def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    df = pd.read_csv(r'/Users/karim/Desktop/Downloads/diabetes.csv', encoding='latin-1', delimiter=",")
    zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
    for column in zero_not_accepted:
        df[column] = df[column].replace(0, np.NaN)
        mean = int(df[column].mean(skipna=True))
        df[column] = df[column].replace(np.NaN, mean)

    X = df.iloc[:, 0:8]
    y = df.iloc[:, 8]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)

    

    log = LogisticRegression(solver="liblinear")
    log_model = log.fit(X_train, y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    pred = log_model.predict([[val1,val2,val3,val4,val5,val6,val7,val8]])

    result1=""
    if pred == [1]:
        result1 = "Positive"
    else:
        result1 = "Négatif"

    return render(request,  'predict.html',{"result2":result1})
