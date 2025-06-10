
from django.urls import path
from .views import chat_with_prompt_template

urlpatterns = [
    path('chatai/', chat_with_prompt_template, name='chat-api'),
]