# from sklearn.preprocessing import StandardScaler
import os

from app import app

from flask import Flask,request,jsonify,render_template

app=Flask(__name__,template_folder='template')

import numpy as np
import joblib


def inp(scale):
    joblib_file = "UsaHouse_Model.pkl"
    model = joblib.load(joblib_file)

    pred = model.predict(scale)
    return pred


@app.route('/')
def correct():
    return render_template('USAinput.html')


@app.route('/USAinput', methods=['POST'])
def success():
    # fname1,2,3 is a acutally a name
    Income = request.form.get("fname1")
    Age = request.form.get("fname2")
    Population = request.form.get("fname3")

    r = inp(np.log(np.array([[Income, Age, Population]], dtype=int)))

    return render_template("USAinput.html", name=r[0])


if __name__ == '__main__':
    app.run()
