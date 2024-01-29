# prediction_app/views.py
import base64

import matplotlib
matplotlib.use('Agg')
from django.shortcuts import render
from django.http import JsonResponse
from keras.preprocessing import image
import numpy as np
from io import BytesIO

from matplotlib import pyplot as plt

from .model import load_prediction_model

def index(request):
    return render(request, 'home.html')

def index2(request):
    # Data
    predictions = [79.31381514736297, 82.55346375125761, 85.44122311238552, 87.9371667092085, 90.15295373280968, 92.43440858187829, 91.74768479768213]


    # Days
    days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7']

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(days, predictions, marker='o', label='Stock Prices', color='b')
    ax.set_title('Stock Prices')
    ax.set_xlabel('Days')
    ax.set_ylabel('Stock Prices')
    ax.grid(True)
    ax.legend()

    # Convert the plot to an image
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    data = base64.b64encode(buf.getvalue()).decode('utf-8')
    img_tag = f'<img src="data:image/png;base64,{data}" alt="Stock Prices">'

    return render(request, 'index2.html', {'img_tag': img_tag})

def index1(request):
    predictions = [118.73303820347446, 123.60354004264636, 128.45739710672657, 132.6155120091262, 136.39333331875116, 140.54667066345263, 144.06489954170465]

    # Days
    days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7']

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(days, predictions, marker='o', label='Stock Prices', color='b')
    ax.set_title('Stock Prices')
    ax.set_xlabel('Days')
    ax.set_ylabel('Stock Prices')
    ax.grid(True)
    ax.legend()

    # Convert the plot to an image
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    data = base64.b64encode(buf.getvalue()).decode('utf-8')
    img_tag = f'<img src="data:image/png;base64,{data}" alt="Stock Prices">'

    return render(request, 'index.html', {'img_tag': img_tag})


import json
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
        return JsonResponse({'error': 'No image uploaded or invalid request.'})
