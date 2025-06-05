
from django.urls import path
from .views import chaat_view

urlpatterns = [
    path('chat/', chaat_view, name='chat-api'),
]