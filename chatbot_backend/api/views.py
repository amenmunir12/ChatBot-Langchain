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
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from rest_framework.parsers import MultiPartParser, FormParser
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings

import os
import tempfile
# class FileUploadView(APIView):
#     parser_classes = (MultiPartParser, FormParser)

#     def post(self, request, format=None):
#         serializer = DocumentSerializer(data=request.data)
#         if serializer.is_valid():
#             document = serializer.save()

#             uploaded_file = request.FILES.get('file')
#             if not uploaded_file:
#                 return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)

#             # Save file metadata
#             document.filename = uploaded_file.name
#             document.content_type = uploaded_file.content_type
#             document.file_size = uploaded_file.size
#             document.save()

#             # Step 1: Save the file temporarily (or use in-memory processing)
#             with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
#                 for chunk in uploaded_file.chunks():
#                     temp_file.write(chunk)
#                 temp_file_path = temp_file.name 
#             os.remove(temp_file_path)

#             # Step 2: Load and split the document
#             loader = self.get_loader(temp_file_path, uploaded_file.content_type)
#             pages = loader.load()
#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#             chunks = text_splitter.split_documents(pages)

#             # Step 3: Initialize Ollama embeddings (local, no API key)
#             embeddings = OllamaEmbeddings(model="nomic-embed-text")  # Or "llama2"

#             # Step 4: Store in ChromaDB (persistent local storage)
#             chroma_db = Chroma.from_documents(
#                 documents=chunks,
#                 embedding=embeddings,
#                 persist_directory=f"chroma_db/{document.id}"  # Unique path per document
#             )
#             chroma_db.persist()  # Ensure data is saved to disk

#             # Cleanup (optional)
#             os.remove(temp_file_path)

#             return Response(DocumentSerializer(document).data, status=status.HTTP_201_CREATED)
#         else:
#             return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# import tempfile
# import os

# class FileUploadView(APIView):
#     parser_classes = (MultiPartParser, FormParser)

#     def post(self, request, format=None):
#         serializer = DocumentSerializer(data=request.data)
#         if serializer.is_valid():
#             document = serializer.save()

#             uploaded_file = request.FILES.get('file')
#             if not uploaded_file:
#                 return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)

#             # Save file metadata
#             document.filename = uploaded_file.name
#             document.content_type = uploaded_file.content_type
#             document.file_size = uploaded_file.size
#             document.save()

#             # Save uploaded file to a temp file on disk
#             suffix = os.path.splitext(uploaded_file.name)[1]
#             with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
#                 for chunk in uploaded_file.chunks():
#                     temp_file.write(chunk)
#                 temp_file_path = temp_file.name

#             try:
#                 # Load and split the document
#                 loader = self.get_loader(temp_file_path, uploaded_file.content_type)
#                 pages = loader.load()
#                 text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#                 chunks = text_splitter.split_documents(pages)

#                 # Embeddings and vector store
#                 embeddings = HuggingFaceEmbeddings(
#                 model_name="sentence-transformers/all-MiniLM-L6-v2",
#                 model_kwargs={"device": "cpu"}
#             )
#                 chroma_db = Chroma.from_documents(
#                     documents=chunks,
#                     embedding=embeddings,
#                     persist_directory="chroma_db"
# )
#                 chroma_db.persist()

#                 return Response(DocumentSerializer(document).data, status=status.HTTP_201_CREATED)

#             finally:
#                 # Always clean up the temporary file
#                 os.remove(temp_file_path)

#         else:
#             return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#     def get_loader(self, file_path, content_type):
#         """Helper to choose the right loader based on file type."""
#         if content_type == "application/pdf":
#             return PyPDFLoader(file_path)
#         elif content_type == "text/plain":
#             return TextLoader(file_path)
#         elif content_type in [
#             "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
#             "application/msword",
#         ]:
#             return Docx2txtLoader(file_path)
#         else:
#             raise ValueError(f"Unsupported file type: {content_type}")

# from django.http import JsonResponse
# import traceback

# class FileUploadView(APIView):
#     parser_classes = (MultiPartParser, FormParser)

#     def post(self, request, format=None):
#         serializer = DocumentSerializer(data=request.data)
#         if serializer.is_valid():
#             document = serializer.save()

#             uploaded_file = request.FILES.get('file')
#             if uploaded_file:
#                 document.filename = uploaded_file.name
#                 document.content_type = uploaded_file.content_type
#                 document.file_size = uploaded_file.size
#                 document.save()

#                 file_path = document.file.path

#                 try:
#                     # === Load PDF ===
#                     loader = PyPDFLoader(file_path)
#                     docs = loader.load()

#                     # === Split Text ===
#                     text_splitter = RecursiveCharacterTextSplitter(
#                         chunk_size=1500,
#                         chunk_overlap=150
#                     )
#                     splits = text_splitter.split_documents(docs)

#                     # === Embedding and Vectorstore ===
                    
#                     embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


#                     persist_directory = os.path.join('docs', 'chroma')

#                     vectordb = Chroma.from_documents(
#                         documents=splits,
#                         embedding=embedding,
#                         persist_directory=persist_directory
#                     )
#                     vectordb.persist()

#                     print("Uploaded file:", uploaded_file.name)
#                     print("Total Chunks:", len(splits))
#                     print("Total vectors in DB:", vectordb._collection.count())

#                 except Exception as e:
#                     traceback_str = traceback.format_exc()
#                     print("Error processing PDF:", traceback_str)
#                     return JsonResponse({'error': 'Error processing PDF', 'details': str(e)}, status=500)

#             return Response(DocumentSerializer(document).data, status=status.HTTP_201_CREATED)
#         else:
#             return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)




# views.py

import os
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

import tempfile
import json


VECTOR_INDEX_PATH = os.path.join(settings.BASE_DIR, "vector_index")

@csrf_exempt
def upload_file(request):
    if request.method == 'POST':
        print("Request content-type:", request.content_type)
        print("request.FILES keys:", request.FILES.keys())
        print("request.POST keys:", request.POST.keys())

        file = request.FILES.get('file')
        if not file:
            return JsonResponse({'error': 'No file provided'}, status=400)

   
        temp_path = default_storage.save('temp/' + file.name, ContentFile(file.read()))
        temp_full_path = os.path.join(settings.MEDIA_ROOT, temp_path)

     
        loader = TextLoader(temp_full_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(docs, embeddings)

   
        vectordb.save_local(VECTOR_INDEX_PATH)

        return JsonResponse({'message': 'File processed and vectorized successfully'})
    else:
        return JsonResponse({'error': 'Invalid method'}, status=405)




from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

@csrf_exempt
def ask_question(request):
    if request.method == 'POST':
        try:
            body = json.loads(request.body)
            query = body.get('query', '')
            if not query:
                return JsonResponse({'error': 'No query provided'}, status=400)

       
            embeddings = OpenAIEmbeddings()
            vectordb = FAISS.load_local(VECTOR_INDEX_PATH, embeddings)

           
            docs = vectordb.similarity_search(query)

          
            llm = OpenAI(temperature=0)
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)

            return JsonResponse({'response': response})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid method'}, status=405)











        
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












