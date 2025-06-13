from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.llm_pipeline import load_pdf_file,text_split,clean_and_tokenize_chunks,download_hugging_face_embeddings

extracted_data = load_pdf_file(r"C:\Users\rithi\Desktop\GEN_AI\chat_bot\data\Medicines_for_Cats_and_Dogs_final.pdf")  
text_chunks = text_split(extracted_data)
cleaned_texts, all_tokens = clean_and_tokenize_chunks(text_chunks)

cleaned_documents = [Document(page_content=text) for text in cleaned_texts]


embeddings = download_hugging_face_embeddings()

db_directory = "./chroma_db"


docsearch = Chroma(
    persist_directory=db_directory,
    embedding_function=embeddings,
)

docsearch.add_documents(cleaned_documents)

docsearch.persist()
















