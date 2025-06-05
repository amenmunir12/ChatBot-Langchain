from django.db import models


class Question(models.Model):
    prompt = models.CharField(max_length=256)
    response = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.prompt