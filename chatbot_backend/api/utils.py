# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import CharacterTextSplitter

# def process_document_with_langchain(file_path):
#     # Load PDF
#     loader = PyPDFLoader(file_path)
#     pages = loader.load()
    
#     # Split into chunks
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=150,
#         length_function=len
#     )
#     chunks = text_splitter.split_documents(pages)
    
#     return chunks
import os
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import CharacterTextSplitter

def process_document_with_langchain(file_path):

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    pages = loader.load()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_documents(pages)

    return chunks

