# views.py

import os
import json
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from chatbot_backend import settings  # change to your actual project name

VECTOR_INDEX_PATH = os.path.join(settings.BASE_DIR, "vector_index")

@csrf_exempt
def chat_with_prompt_template(request):
    if request.method == 'POST':
        try:
            body = json.loads(request.body)
            user_query = body.get('message', '')
            if not user_query:
                return JsonResponse({'error': 'Query message is required'}, status=400)

            # Load vector store
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectordb = FAISS.load_local(VECTOR_INDEX_PATH, embeddings)

            # Perform similarity search
            docs = vectordb.similarity_search(user_query, k=4)
            context = "\n\n".join([doc.page_content for doc in docs])

            # Define prompt template
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

            # Use LLM with the custom prompt
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
