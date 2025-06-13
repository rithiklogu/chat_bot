from src.llm_pipeline import load_pdf_file, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Load and process the PDF data
extracted_data = load_pdf_file(data="Data/")
text_chunks = text_split(extracted_data)

# Load Hugging Face embeddings
embeddings = download_hugging_face_embeddings()

# Define the local storage directory for ChromaDB
db_directory = "./chroma_db"

# Create or load the Chroma vector store
docsearch = Chroma(
    persist_directory=db_directory,
    embedding_function=embeddings,
)

# Add documents to ChromaDB
docsearch.add_documents(text_chunks)

# Save the vector store (ensures persistence across restarts)
docsearch.persist()
