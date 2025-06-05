from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .key import keychatgpt
from .utils import process_document_with_langchain
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
            document = serializer.save()

            uploaded_file = request.FILES.get('file')
            if uploaded_file:
                document.filename = uploaded_file.name
                document.content_type = uploaded_file.content_type  
                document.file_size = uploaded_file.size
                document.save()
                file_path = document.file.path
                chunks = process_document_with_langchain(file_path)

            # OPTIONAL: Store chunks into DB or Vector Store (next step)
                print("Total Chunks:", len(chunks))  # Debugging output

            return Response(DocumentSerializer(document).data, status=status.HTTP_201_CREATED)
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


import os
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

ENDPOINT = "https://models.github.ai/inference"
MODEL = "openai/gpt-4.1"
TOKEN = "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"
if not TOKEN:
    raise RuntimeError(" environment variable not set")

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

    userMessage = data.get('message')
    if not userMessage:
        return JsonResponse({'error': 'No message provided'}, status=400)

    try:
        response = client.complete(
            messages=[
                SystemMessage("test msg"),
                UserMessage(userMessage),
            ],
            temperature=1,
            top_p=1,
            model=MODEL,
        )
        answer = response.choices[0].message.content
        return JsonResponse({'response': answer})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)












