from flask import Flask,request,jsonify,render_template
import numpy as np
import pickle

app=Flask(__name__,template_folder='template')


joblib_file = "UsaHouse_Model.pkl"
model = pickle.load(open(joblib_file,'rb'))

@app.route('/')
def correct():
    return render_template('USAinput.html')


@app.route('/USAinput', methods=['POST'])
def success():
    # fname1,2,3 is a acutally a name
    Income = request.form.get("fname1")
    Age = request.form.get("fname2")
    Population = request.form.get("fname3")

    scale= (np.log(np.array([[Income, Age, Population]], dtype=int)))
    pred = model.predict(scale)
    return render_template("USAinput.html", name=pred[0])


if __name__ == '__main__':
    app.run(debug=True)

