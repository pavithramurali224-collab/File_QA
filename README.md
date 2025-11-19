# File_QA
This documentation explains the full workflow for building a Retrieval-Augmented Generation (RAG) system using LangChain, FAISS vector store, HuggingFace embeddings, and a FLAN-T5 model. The script allows uploading documents (PDF, DOCX, or TXT), converting them into chunks, embedding them, storing them in FAISS.

1. Installing Required Libraries
Run the following commands in Google Colab to install required dependencies:
!pip install langchain langchain-community
!pip install faiss-cpu
!pip install pypdf python-docx
!pip install sentence-transformers
!pip install transformers
2. Uploading Files
Users upload files using Google Colab's file upload interface:
from google.colab import files
uploaded = files.upload()
file_path = list(uploaded.keys())[0]
print('Uploaded:', file_path)
3. Document Loading
The script automatically detects the file format and uses the correct loader:
- PDF → PyPDFLoader
- DOCX → Docx2txtLoader
- TXT → TextLoader
4. Chunking the Text
The document is split into smaller chunks using RecursiveCharacterTextSplitter:
- chunk_size = 500
- chunk_overlap = 100
This improves retrieval accuracy and ensures smooth embedding.
5. Generating Embeddings & Creating FAISS Index
Embeddings are generated using MiniLM:
Model: sentence-transformers/all-MiniLM-L6-v2
The embeddings are stored in a FAISS vector database for fast similarity search.
6. Loading the Language Model
A lightweight text-generation model is used:
- Model: google/flan-t5-base
This model is wrapped using HuggingFacePipeline.
7. Building the RetrievalQA Chain
RetrievalQA combines vector search and the language model to respond to user queries. It retrieves the most relevant chunks and passes them to the model to generate an answer.
8. Running Queries
Users can ask questions using both a test query and an interactive loop. The loop continues until the user enters 'exit'.
9. Conclusion
This script provides a complete implementation of a simple yet powerful RAG system that can be expanded with more advanced models, better embeddings, and deployment options.
