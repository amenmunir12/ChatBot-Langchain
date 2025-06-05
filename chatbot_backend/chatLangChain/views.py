import json
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from langchain.llms import OpenAI
from langchain.chains import ConversationChain



@csrf_exempt
def chaat_view(request):
    pass
  
