from django.urls import path
from . import views

app_name = 'predictor'

urlpatterns = [
    path('', views.predict_image, name='predict_image'),  # home page + upload
]
