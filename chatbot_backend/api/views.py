from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json


from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from .serializers import DocumentSerializer

class FileUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, format=None):
        serializer = DocumentSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@csrf_exempt
def chat_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        message = data.get('message', '')
        response = f"Bot says: You said '{message}'"
        return JsonResponse({'response': response})
    return JsonResponse({'error': 'Only POST method allowed'}, status=400)


# import openai
# openai.api_key = "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"
# @csrf_exempt
# def chat_with_openai(request):
#     if request.method == 'POST':
#         data = json.loads(request.body)
#         message = data.get('message', '')
        
#         if not message:
#             return JsonResponse({'error': 'No message provided'}, status=400)

#         try:
#             response = openai.ChatCompletion.create(
#                 model="gpt-3.5-turbo",
#                 messages=[{"role": "user", "content": message}]
#             )
#             answer = response.choices[0].message['content']
#             return JsonResponse({'response': answer})
#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)

#     return JsonResponse({'error': 'Only POST method allowed'}, status=400)
# import json
# import openai
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt

# from openai import OpenAI

# client = OpenAI(api_key="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")  # Preferably use environment variable

# @csrf_exempt
# def chat_with_openai(request):
#     if request.method == 'POST':
#         data = json.loads(request.body)
#         message = data.get('message', '')

#         if not message:
#             return JsonResponse({'error': 'No message provided'}, status=400)

#         try:
#             response = client.chat.completions.create(
#                 model="gpt-4.1",
#                 messages=[{"role": "user", "content": message}]
#             )
#             answer = response.choices[0].message.content
#             return JsonResponse({'response': answer})
#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)

#     return JsonResponse({'error': 'Only POST method allowed'}, status=400)

# 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
import os
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# Setup your endpoint, model, and token
ENDPOINT = "https://models.github.ai/inference"
MODEL = "openai/gpt-4.1"
TOKEN = "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"  # Make sure this env var is set

if not TOKEN:
    raise RuntimeError("GITHUB_TOKEN environment variable not set")

client = ChatCompletionsClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(TOKEN),
)

@csrf_exempt
def chat_gpt(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)

    if not request.body:
        return JsonResponse({'error': 'Empty request body'}, status=400)

    try:
        data = json.loads(request.body.decode('utf-8'))
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    user_message = data.get('message')
    if not user_message:
        return JsonResponse({'error': 'No message provided'}, status=400)

    try:
        response = client.complete(
            messages=[
                SystemMessage("You are a helpful assistant."),
                UserMessage(user_message),
            ],
            temperature=1,
            top_p=1,
            model=MODEL,
        )
        answer = response.choices[0].message.content
        return JsonResponse({'response': answer})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)












