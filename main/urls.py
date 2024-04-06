from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),      # URL for the index view
    path('calculate/', views.results, name='calculate'),  # URL for the calculate view
]
