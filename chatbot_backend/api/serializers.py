from rest_framework import serializers
from .models import Document

class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ['id', 'file', 'uploaded_at', 'filename', 'content_type', 'file_size']
        read_only_fields = ['uploaded_at', 'filename', 'content_type', 'file_size']
