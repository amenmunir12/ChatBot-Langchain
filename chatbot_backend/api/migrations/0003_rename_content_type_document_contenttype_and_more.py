# Generated by Django 5.2.1 on 2025-06-05 11:59

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0002_document_content_type_document_file_size_and_more'),
    ]

    operations = [
        migrations.RenameField(
            model_name='document',
            old_name='content_type',
            new_name='contentType',
        ),
        migrations.RenameField(
            model_name='document',
            old_name='file_size',
            new_name='fileSize',
        ),
        migrations.RenameField(
            model_name='document',
            old_name='uploaded_at',
            new_name='uploadedAt',
        ),
    ]
