from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader

file_path = r"data\Medicines_for_Cats_and_Dogs_final.pdf"
loader = PyPDFLoader(file_path)
pages = []
async for page in loader.alazy_load():
    pages.append(page)