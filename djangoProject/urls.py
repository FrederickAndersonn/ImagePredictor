# prediction_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    path('index2/', views.index2, name='index2'),  # Add this line
    path('index1/', views.index1, name='index1'),  # Add this line
]
