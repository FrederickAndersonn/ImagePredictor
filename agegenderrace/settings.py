# age_gender_race_prediction/settings.py
import os

from djangoProject.settings import BASE_DIR

INSTALLED_APPS = [
    # ...
    'djangoProject',
]

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'prediction_app', 'templates')],
        # ...
    },
]
