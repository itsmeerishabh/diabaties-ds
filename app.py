import pickle
from flask import Flask,request,jsonify,render_template
from flask import Response
import numpy 
import pandas
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


LR_model     = pickle.load(open("pickle/logistic_regression_file_diabaties.pkl","rb"))
standard_scaler = pickle.load(open("pickle/scaler_file_diabaties.pkl","rb"))


#route for home page
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict",methods = ["GET","POST"])
def predict_datapoint():
    if request.method=='POST': #after submit the form
        Pregnancies=float(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))
       

        new_data_scaled=standard_scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        result=LR_model.predict(new_data_scaled)

        return render_template('home.html',result=result[0])

    else:
        return render_template('home.html')  #for the GET request in get reqst only page will view so in start call fum 
        #this line will work




if __name__=="__main__":
    app.run(host="0.0.0.0",port= 5001)
