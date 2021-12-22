from django.urls import path
from application import views

urlpatterns = [
    
    path('', views.index),
    path('check_review', views.check_review)
    
]