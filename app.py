import os
import cv2
import numpy as np
import pandas as pd
from flask import Flask, request, redirect, render_template, send_from_directory

import eyecrop
import tensorflow as tf

COUNT = 0
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    global COUNT
    if request.method == 'POST':
        if request.files:
            img = request.files['image']
            img.save('static/uploadedthis.jpg'.format(COUNT))
            #file.save(os.path.join(app.config["IMAGE_UPLOADS"], file.filename))

        # Process uploaded image
        L_eye, R_eye = eyecrop.process_face_img('static/uploadedthis.jpg'.format(COUNT))

        # Load in saved model
        saved_model = tf.keras.models.load_model('cnnCat2.h5')
        L_eye_array = np.asarray(L_eye)

        L_eye_array = L_eye_array.reshape(-1, 24, 24,1)

        R_eye_array = np.asarray(R_eye)

        R_eye_array = R_eye_array.reshape(-1, 24, 24, 1)

        left_eye_pred = saved_model.predict(L_eye_array)[0][0].round()
        right_eye_pred = saved_model.predict(L_eye_array)[0][0].round()

        final_preds = eyecrop.convert_words([left_eye_pred, right_eye_pred])

        eyecrop.save_cropped_eyes(
            '{}'.format(COUNT), L_eye, R_eye, final_preds[0], final_preds[1])

        files = eyecrop.get_file_logs('upload/crop')
        COUNT += 1
        return render_template('sub.html', data=[final_preds[0], files[1]])

@app.route('/load_img/<path:filename>')
def load_img(filename):
    global COUNT
    return send_from_directory('upload.jpg', filename)

if __name__ == '__main__':
    app.run(debug=True)