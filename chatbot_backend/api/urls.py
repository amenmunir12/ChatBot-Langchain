# urls.py
from django.urls import path
from .views import FileUploadView , chat_view,chat_gpt

urlpatterns = [
    path('upload/', FileUploadView.as_view(), name='file-upload'),
    path('chat/', chat_view, name='file-upload'),
    path('chat_with_openai/', chat_gpt, name='chat-with-openai'),
]
