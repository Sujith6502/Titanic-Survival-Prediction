#!pip install sklearn xgboost
import pandas as pd
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score

df = pd.read_csv('D:\\Titanic_survival\\titanic.csv')
X = df.drop('Survived', axis=1) 
y = df.Survived

print(df)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.1, random_state=1234)

class PrepProcesor(BaseEstimator, TransformerMixin): 
    def fit(self, X, y=None): 
        self.ageImputer = SimpleImputer()
        self.ageImputer.fit(X[['Age']])        
        return self 
        
    def transform(self, X, y=None):
        X['Age'] = self.ageImputer.transform(X[['Age']])
        X['CabinClass'] = X['Cabin'].fillna('M').apply(lambda x: str(x).replace(" ", "")).apply(lambda x: re.sub(r'[^a-zA-Z]', '', x))
        X['CabinNumber'] = X['Cabin'].fillna('M').apply(lambda x: str(x).replace(" ", "")).apply(lambda x: re.sub(r'[^0-9]', '', x)).replace('', 0) 
        X['Embarked'] = X['Embarked'].fillna('M')
        X = X.drop(['PassengerId', 'Name', 'Ticket','Cabin'], axis=1)
        return X

preproc = PrepProcesor()
numeric_pipeline = Pipeline([('Scaler', StandardScaler())]) 
categorical_pipeline = Pipeline([('OneHot', OneHotEncoder(handle_unknown='ignore'))])
transformer = ColumnTransformer([('num', numeric_pipeline, ['Pclass','Age','SibSp','Parch','Fare','CabinNumber']), ('cat', categorical_pipeline, ['Sex','Embarked','CabinClass'])])

mlpipe = Pipeline([('InitialPreproc', PrepProcesor()), ('Transformer',transformer), ('xgb', XGBClassifier())])

mlpipe.fit(X_train,y_train)

yhat = mlpipe.predict(X_test) 

precision_score(y_test, yhat) 

recall_score(y_test, yhat) 

import joblib

joblib.dump(mlpipe, 'xgbpipe.joblib') 

model = joblib.load('xgbpipe.joblib')

test = pd.read_csv('D:\\Titanic_survival\\test.csv')


yhat = model.predict(test)

columns = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch','Ticket', 'Fare', 'Cabin', 'Embarked']

import streamlit as st


import numpy as np
import pandas as pd
import joblib

model = joblib.load('xgbpipe.joblib')
st.title('Did they survive? :ship:')
# PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
st.write('''prediction of a passenger **survived** or not based on titanic dataset''')
passengerid = st.text_input("Input Passenger ID", '123456') 
pclass = st.selectbox("Choose class", [1,2,3])
name  = st.text_input("Input Passenger Name", 'John Smith')
sex = st.select_slider("Choose sex", ['male','female'])
age = st.slider("Choose age",0,100)
st.write("you have seleted",age)
sibsp = st.slider("Choose siblings",0,10)
parch = st.slider("Choose parch",0,2)
ticket = st.text_input("Input Ticket Number", "12345") 
fare = st.number_input("Input Fare Price", 0,1000)
cabin = st.text_input("Input Cabin", "C52") 
embarked = st.select_slider("Did they Embark?", ['S','C','Q'])

def predict(): 
    row = np.array([passengerid,pclass,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked]) 
    X = pd.DataFrame([row], columns = columns)
    prediction = model.predict(X)
    if prediction[0] == 1: 
        st.success('Passenger Survived :thumbsup:')
    else: 
        st.error('Passenger did not Survive :thumbsdown:') 

trigger = st.button('Predict', on_click=predict)