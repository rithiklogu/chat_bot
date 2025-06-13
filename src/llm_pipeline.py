from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


# #Extract Data From the PDF File
# def load_pdf_file(data):
#     loader= DirectoryLoader(data,
#                             glob="*.pdf",
#                             loader_cls=PyPDFLoader)

#     documents=loader.load()

#     return documents



# #Split the Data into Text Chunks
# def text_split(extracted_data):
#     text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
#     text_chunks=text_splitter.split_documents(extracted_data)
#     return text_chunks



# #Download the Embeddings from HuggingFace 
# def download_hugging_face_embeddings():
#     embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
#     return embeddings


import fitz  # PyMuPDF
import re
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# Step 1: Extract text from PDF
def load_pdf_file(pdf_path):
    doc = fitz.open(pdf_path)
    text_list = [Document(page_content=page.get_text()) for page in doc]
    doc.close()
    return text_list


# Step 2: Split text into chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# Load BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# Step 3: Clean and tokenize each chunk
def clean_and_tokenize_chunks(text_chunks):
    all_tokens = []
    cleaned_texts = []

    for doc in text_chunks:
        text = doc.page_content.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)  
        cleaned_texts.append(text)

        tokens = tokenizer.tokenize(text)
        all_tokens.append(tokens)

    return cleaned_texts, all_tokens


# Step 4: Load Hugging Face sentence transformer embeddings
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings
