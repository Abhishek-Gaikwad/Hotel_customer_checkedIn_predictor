# -*- coding: utf-8 -*-
"""
@author: ABHISHEK
"""

from flask import Flask, render_template,request
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods =['POST'])
def predict_checkedin():
    ID = int(request.form.get('ID'))
    Age = int(request.form.get('Age'))
    DaysSinceCreation = int(request.form.get('DaysSinceCreation'))
    AverageLeadTime = float(request.form.get('AverageLeadTime'))
    LodgingRevenue = float(request.form.get('LodgingRevenue'))
    OtherRevenue = float(request.form.get('OtherRevenue'))
    BookingsCanceled = float(request.form.get('BookingsCanceled'))
    BookingsNoShowed = float(request.form.get('BookingsNoShowed'))
    PersonsNights = int(request.form.get('PersonsNights'))
    RoomNights = int(request.form.get('RoomNights'))
    DaysSinceLastStay = int(request.form.get('DaysSinceLastStay'))
    DaysSinceFirstStay = int(request.form.get('DaysSinceFirstStay'))
    
    
    #prediction
    result = model.predict(np.array([ID,Age,DaysSinceCreation,AverageLeadTime,LodgingRevenue,OtherRevenue,BookingsCanceled,BookingsNoShowed,PersonsNights,RoomNights,DaysSinceLastStay,DaysSinceFirstStay]).reshape(1,12))
    
    if result[0] >= 1:
        result = 'Customer is going to checked IN'
    else:
        result = 'Customer is not going to checked IN'
    
    return result



if __name__ == '__main__':
    app.run(debug=True)
    
    
