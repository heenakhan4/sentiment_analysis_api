from django.urls import path
from .views import *

urlpatterns = [
    path('register/', register, name='register'),
    path('sentiment/analyze/', Analyze.as_view(), name='analyze'),
]