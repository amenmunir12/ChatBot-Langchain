from django.db import models


from django.db import models

class Document(models.Model):
    file = models.FileField(upload_to='uploads/')  
    uploaded_at = models.DateTimeField(auto_now_add=True)
    filename = models.CharField(max_length=255, blank=True)
    content_type = models.CharField(max_length=100, blank=True)
    file_size = models.PositiveIntegerField(null=True, blank=True)

    def __str__(self):
        return self.file.name


from django.db import models
import json

class DocumentChunk(models.Model):
    chunk_text = models.TextField()
    embedding = models.TextField()  # Store as JSON string
    source_file = models.CharField(max_length=255, blank=True, null=True)

    def set_embedding(self, vector):
        self.embedding = json.dumps(vector)

    def get_embedding(self):
        return json.loads(self.embedding)

    def __str__(self):
        return self.chunk_text[:50] + "..."
