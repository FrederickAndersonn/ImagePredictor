# prediction_app/views.py
import base64

import matplotlib

matplotlib.use('Agg')
from keras.preprocessing import image
import numpy as np
from io import BytesIO

from matplotlib import pyplot as plt

from .model import load_prediction_model


def index(request):
    return render(request, 'home.html')

from django.shortcuts import render


def predict(request):
    if request.method == 'POST' and request.FILES['image']:
        # Assuming you have a form with the name 'image' for file input
        uploaded_file = request.FILES['image']

        # Convert InMemoryUploadedFile to BytesIO
        img_bytes = BytesIO(uploaded_file.read())

        # Load the image using Keras
        img = image.load_img(img_bytes, target_size=(198, 198))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Load the prediction model
        model = load_prediction_model()

        # Make predictions
        age, race, gender = model.predict(img_array)

        # Convert the results to human-readable format
        gender_label = 'Male' if gender[0][0] > 0.5 else 'Female'
        race_label = ['White', 'Black', 'Asian', 'Indian', 'Others'][np.argmax(race)]
        result = {'age': int(age[0][0] * 65), 'gender': gender_label, 'race': race_label}

        return render(request, 'home.html', {'result': result})
    else:
        return
