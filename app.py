#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import SGD
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()

# Load your trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
def load_model1():
    model = load_model('./models/model.h5')
    model.compile(loss = "binary_crossentropy", optimizer = SGD(lr=0.001, momentum=0.9), metrics=["accuracy"])
    print('Model loaded. Check http://localhost:5000/')
    return model


def model_predict(img_path, model):
    print('hello')
    img = image.load_img(img_path, target_size=(64,64))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        f.save(f.filename)
        model = load_model1()
        # Make prediction
        preds = model_predict(f.filename, model)
        print(preds)
        if preds[0][0] > 0.90:
            prediction = "Test Case is Negative"
        elif preds[0][1] > 0.90:
            prediction = "Test Case is Positive"
        else:
            prediction = "UNRECOGNISED"
        return prediction
    return None


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

