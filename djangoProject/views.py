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


def index1(request):
    # Data
    predictions = [147.4199981689453, 149.97000122070312, 154.07000732421875, 153.7899932861328,
                   152.1199951171875, 153.83999633789062, 153.4199981689453, 153.41000366210938,
                   153.33999633789062, 153.3800048828125, 151.94000244140625, 149.92999267578125,
                   148.47000122070312, 144.57000732421875, 145.24000549316406, 149.10000610351562,
                   151.3699951171875, 153.72999572753906, 155.17999267578125, 154.6199951171875,
                   153.16000366210938, 151.7100067138672, 153.5, 155.33999633789062, 154.77999877929688,
                   156.02000427246094, 156.8699951171875, 157.75, 159.1199951171875, 160.05580139160156,
                   136.41544965185196, 138.84944492903566, 141.27682301124614, 143.683927606844, 146.05699139742856, 148.37594074455754, 150.69077099884245]

    days = ['{}'.format(i) for i in range(1, 38)]

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(days[:-7], predictions[:-7], marker='o', label='Stock Prices', color='b')  # Plotting all days except last 7
    ax.plot(days[-8:], predictions[-8:], marker='o', label='Last 7 Days',color='r')  # Plotting last 8 days including the overlap 7 days in red
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


def index2(request):
    predictions = [ 198.11000061035156,
                   197.57000732421875,
                   195.88999938964844,
                   196.94000244140625,
                   194.8300018310547,
                   194.67999267578125,
                   193.60000610351562,
                   193.0500030517578,
                   193.14999389648438,
                   193.5800018310547,
                   192.52999877929688,
                   185.63999938964844,
                   184.25,
                   181.91000366210938,
                   181.17999267578125,
                   185.55999755859375,
                   185.13999938964844,
                   186.19000244140625,
                   185.58999633789062,
                   185.9199981689453,
                   183.6300048828125,
                   182.67999267578125,
                   188.6300048828125,
                   191.55999755859375,
                   193.88999938964844,
                   195.17999267578125,
                   194.5,
                   194.1699981689453,
                   192.4199981689453,
                   190.9499969482422,
                   154.75795250164296, 158.4245451904672, 162.26950003337006, 166.28014765512134, 170.4478530625474,
                   174.89004980352212, 179.9969640722862]

    # Days
    days = ['{}'.format(i) for i in range(1, 38)]

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(days[:-7], predictions[:-7], marker='o', label='Stock Prices', color='b')  # Plotting all days except last 7
    ax.plot(days[-8:], predictions[-8:], marker='o', label='Last 7 Days',color='r')  # Plotting last 8 days including the overlap # Last 7 days in red
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
