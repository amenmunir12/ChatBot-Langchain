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

# @csrf_exempt
# def upload_file(request):
#     if request.method == 'POST':
#         print("Request content-type:", request.content_type)
#         print("request.FILES keys:", request.FILES.keys())
#         print("request.POST keys:", request.POST.keys())

#         file = request.FILES.get('file')
#         if not file:
#             return JsonResponse({'error': 'No file provided'}, status=400)

   
#         temp_path = default_storage.save('temp/' + file.name, ContentFile(file.read()))
#         temp_full_path = os.path.join(settings.MEDIA_ROOT, temp_path)

     
#         loader = TextLoader(temp_full_path)
#         documents = loader.load()

#         splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#         docs = splitter.split_documents(documents)

#         embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         vectordb = FAISS.from_documents(docs, embeddings)

   
#         vectordb.save_local(VECTOR_INDEX_PATH)

#         return JsonResponse({'message': 'File processed and vectorized successfully'})
#     else:
#         return JsonResponse({'error': 'Invalid method'}, status=405)



from .models import DocumentChunk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
from django.conf import settings

@csrf_exempt
def upload_file(request):
    if request.method == 'POST':
        file = request.FILES.get('file')
        if not file:
            return JsonResponse({'error': 'No file provided'}, status=400)

     
        temp_path = default_storage.save('temp/' + file.name, ContentFile(file.read()))
        temp_full_path = os.path.join(settings.MEDIA_ROOT, temp_path)

        loader = TextLoader(temp_full_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(documents)

        
        embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        for doc in docs:
            embedding = embeddings_model.embed_query(doc.page_content)

            chunk = DocumentChunk(
                chunk_text=doc.page_content,
                source_file=file.name
            )
            chunk.set_embedding(embedding)
            chunk.save()

        return JsonResponse({'message': 'Chunks stored in SQLite DB successfully'})
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











        
# @csrf_exempt
# def chat_view(request):
#     if request.method == 'POST':
#         data = json.loads(request.body)
#         message = data.get('message', '')
#         response = f"Bot says: You said '{message}'"
#         return JsonResponse({'response': response})
#     return JsonResponse({'error': 'Only POST method allowed'}, status=400)


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









# this view has the error where it wont let me axessx the open ai gpt model qithput api key and my key stays beng out of quota 
# i have this above fun written where i used github tokens to use the gpt model for free but i am having issues to use the same approcah ere 
import os
import json
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from chatbot_backend import settings 

VECTOR_INDEX_PATH = os.path.join(settings.BASE_DIR, "vector_index")

@csrf_exempt
def chat_with_prompt_template(request):
    if request.method == 'POST':
        try:
            body = json.loads(request.body)
            user_query = body.get('message', '')
            if not user_query:
                return JsonResponse({'error': 'Query message is required'}, status=400)

         
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectordb = FAISS.load_local(VECTOR_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)


            
            docs = vectordb.similarity_search(user_query, k=4)
            context = "\n\n".join([doc.page_content for doc in docs])
            print("Retrieved docs:", docs)
            print("Context:", context)
            



            
            template = """
You are a helpful assistant. Use the following document excerpts to answer the user's question.
If the answer is not in the provided content, say you don't know. Be concise and clear.

Context:
{context}

Question: {question}
Answer:
"""

            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=template,
            )

            
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
            chain = LLMChain(llm=llm, prompt=prompt)

            response = chain.run({
                "context": context,
                "question": user_query
            })

            return JsonResponse({'response': response})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)

