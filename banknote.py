from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in= open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)
@app.route('/')
def welcome():
    return "WELCOME TO THE BLANCO WORLD WHERE NOTHING IS REALLY PRETTY "

@app.route('/predict/')
def predict_note():
    variance= request.args.get('variance')
    skewness= request.args.get('skewness')
    curtosis= request.args.get('curtosis')
    entropy= request.args.get('entropy')
    prediction= classifier.predict([[variance, skewness, curtosis, entropy]])
    return "The predicted value is" + str(prediction)
    if __name__=='__main__':
        app.run()
    
if __name__=='__main__':
    app.run()

@app.route('/predict_form',methods=["GET"])
def predict_note_authentication(variance,skewness,curtosis,entropy):
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return prediction)
@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    df_test=pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction=classifier.predict(df_test)
    return str(list(prediction))





