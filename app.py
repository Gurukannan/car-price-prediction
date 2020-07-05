#Importing necessary packages
import numpy as np
from flask import Flask, request, render_template
import pickle
from fastai.tabular import *
import os

#Saving the working directory and model directory
cwd = os.getcwd()
path = cwd + '/model'

#Initializing the FLASK API
app = Flask(__name__)

#Loading the saved model using fastai's load_learner method
model = load_learner(path, 'model.pkl')

#Defining the home page for the web service
@app.route('/')
def home():
    return render_template('index.html')

#Writing api for inference using the loaded model
@app.route('/predict',methods=['POST'])

#Defining the predict method get input from the html page and to predict using the trained model

def predict():
    
    try:
    	#all the input labels . We had only trained the model using these selected features.
        labels = ['Brand', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type','Transmission', 'Owner_Type', 'Mileage']
        
        #Collecting values from the html form and converting into respective types as expected by the model
        Brand =  request.form["Brand"]
        Location =  request.form["Location"]
        Year = int(request.form["Year"])
        KMD =  int(request.form["Kilometers_Driven"])
        Fuel_type = request.form["Fuel_Type"]
        Transmission = request.form['Transmission']
        Owner_Type =  request.form["Owner_Type"]
        Mileage = float(request.form["Mileage"])

        #making a list of the collected features
        features = [Brand , Location , Year, KMD, Fuel_type, Transmission, Owner_Type, Mileage]

        #fastai predicts from a pandas series. so converting the list to a series
        to_predict = pd.Series(features, index = labels)

        #Getting the prediction from the model and rounding the float into 2 decimal places
        prediction = round(float(model.predict(to_predict)[1]),2)

        # Making all predictions below 0 lakhs and above 200 lakhs as invalid
        if prediction > 0 and prediction <= 200:
            return render_template('index.html', prediction_text='Your Input : {} Resale Cost: {} Lakh Rupees'.format(features,prediction))
        else:
            return render_template('index.html', prediction_text='Invalid Prediction !! Network Unable To Predict For The Given Inputs')

    except:
        return render_template('index.html', prediction_text='Prediction Err !!!')

if __name__ == "__main__":
    app.run(debug=True)