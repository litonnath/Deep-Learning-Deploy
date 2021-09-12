from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from PIL import Image

import numpy as np

from flask import Flask,request,jsonify,render_template
import os
tmdir=os.path.abspath('C:\\Users\\Liton\\PycharmProjects\\pythonProject\\tutorial\\tutorial\\spiders')
app=Flask(__name__,template_folder=tmdir)

def load_image(img_pa):
    img = Image.open(img_pa)
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    resized_image = cv2.resize(gray, (124, 124))

    resized_image = resized_image.reshape(124, 124, 1)

    h1 = np.expand_dims(resized_image, axis=0)
    h1 = h1 / 255.0

    plt.figure()

    plt.imshow(h1[0])

    plt.show()

    best_model = load_model('best_model.h5')

    pred = best_model.predict(h1)
    return np.where(pred > 0.5, 'Dog', 'Cat')[0][0]


@app.route('/')
def correct():
    return render_template('div.html')


@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        r = load_image(f)

        return render_template("success.html", name=r)


if __name__ == '__main__':
    app.run(debug=True)
