# urls.py
from django.urls import path
from .views import chat_view,chat_gpt,upload_file, ask_question,chat_with_prompt_template

urlpatterns = [
    # path('upload/', FileUploadView.as_view(), name='file-upload'),
    path('chat/', chat_view, name='file-upload'),
    path('chat_with_openai/', chat_gpt, name='chat-with-openai'),
    path('upload/', upload_file, name='upload_file'),
    path('ask/', chat_with_prompt_template, name='ask_question'),
    path('chat_with_template/', chat_with_prompt_template, name='chat_with_template'),
]
