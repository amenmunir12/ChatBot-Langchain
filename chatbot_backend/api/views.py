
import os
import tempfile
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import json


VECTOR_INDEX_PATH = os.path.join(settings.BASE_DIR, "vector_index")

from .models import DocumentChunk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader

from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from django.conf import settings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings


@csrf_exempt
def upload_file(request):
    if request.method == 'POST':
        file = request.FILES.get('file')
        if not file:
            return JsonResponse({'error': 'No file provided'}, status=400)

        temp_path = default_storage.save('temp/' + file.name, ContentFile(file.read()))
        temp_full_path = os.path.join(settings.MEDIA_ROOT, temp_path)

        try:
           
            loader = PyPDFLoader(temp_full_path)
            documents = loader.load()
        except Exception as e:
            return JsonResponse({'error': f'Failed to load file: {str(e)}'}, status=500)

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(documents)

       
        embeddings_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  
        )

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
        return JsonResponse({'error': 'Invalid method'}, status=40


                            

import os
import json
import requests
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.base import LLM
from typing import Optional, List

from chatbot_backend import settings 

VECTOR_INDEX_PATH = os.path.join(settings.BASE_DIR, "vector_index")


class LocalLLM(LLM):
    endpoint: str = "http://192.168.1.13:1234/v1/chat/completions"  
    model_name: str = "opengpt-3"  
    temperature: float = 0.7

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
        }

        try:
            response = requests.post(self.endpoint, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            # print("Raw LLM Response:", data) 
            # print("Full LLM JSON:", json.dumps(data, indent=2))

            
            if "choices" in data:
                if "message" in data["choices"][0]:
                    return data["choices"][0]["message"]["content"].strip()
                elif "text" in data["choices"][0]:
                    return data["choices"][0]["text"].strip()
                else:
                    return "Unexpected response format in 'choices'"
            elif "output" in data:
                return data["output"].strip()
            else:
                return "Unexpected response format"

        except Exception as e:
            return f"Error communicating with local LLM: {str(e)}"

    @property
    def _identifying_params(self):
        return {"model_name": self.model_name}

    @property
    def _llm_type(self):
        return "local_llm"


@csrf_exempt
def chatWithPromptTemplate(request):
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
            print(f"context: {context}")  

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

            llm = LocalLLM()
            chain = LLMChain(llm=llm, prompt=prompt)

            response = chain.run({
                "context": context,
                "question": user_query
            })

            return JsonResponse({'response': response})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)
