# urls.py
from django.urls import path
from .views import upload_file,chatWithPromptTemplate

urlpatterns = [
    # path('upload/', FileUploadView.as_view(), name='file-upload'),
 
    # path('chat_with_openai/', chat_gpt, name='chat-with-openai'),
    path('upload/', upload_file, name='upload_file'),
    path('ask/', chatWithPromptTemplate , name='ask_question'),
    path('chat_with_template/',chatWithPromptTemplate, name='chat_with_template'),
]
