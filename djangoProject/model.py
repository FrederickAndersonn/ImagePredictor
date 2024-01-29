# prediction_app/model.py
from keras.models import load_model
import os


def load_prediction_model():
    # Provide the correct filename for age_gender_race_model.h5
    model_filename = 'age_gender_race_model.h5'
    model_path = os.path.join(os.path.dirname(__file__), model_filename)

    # Check if the file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file {model_filename} was not found at {model_path}")

    return load_model(model_path)
