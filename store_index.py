from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.llm_pipeline import load_pdf_file,text_split,clean_and_tokenize_chunks,download_hugging_face_embeddings

# Load and process the PDF
extracted_data = load_pdf_file(r"C:\Users\rithi\Desktop\GEN_AI\chat_bot\data\Medicines_for_Cats_and_Dogs_final.pdf")  
text_chunks = text_split(extracted_data)
cleaned_texts, all_tokens = clean_and_tokenize_chunks(text_chunks)

# Convert cleaned texts to LangChain Document objects
cleaned_documents = [Document(page_content=text) for text in cleaned_texts]

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Define ChromaDB directory
db_directory = "./chroma_db"

# Initialize Chroma vector store
docsearch = Chroma(
    persist_directory=db_directory,
    embedding_function=embeddings,
)

# Add cleaned documents (not raw, not tokens)
docsearch.add_documents(cleaned_documents)

# Persist the vector DB
docsearch.persist()
















