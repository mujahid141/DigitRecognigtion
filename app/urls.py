from django.urls import path
from . import views

urlpatterns = [
    path('predict_digit/',views.predict_digit , name='predict_digit'),  # Add this line
]
