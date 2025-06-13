from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os

# ✅ Updated imports based on LangChain deprecations
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ API Key. Check your .env file.")

# Initialize Flask app
app = Flask(__name__)

# Step 1: Load embeddings & Chroma vector store
db_directory = "chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
docsearch = Chroma(persist_directory=db_directory, embedding_function=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# Step 2: Load LLaMA 3 model from Groq
llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.5, max_tokens=500)

# Step 3: Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an assistant for question-answering tasks. "
     "Use the following pieces of retrieved context to answer the question. "
     "If you don't know the answer, say 'This is out of context, I am a helpful pet's (Cat and Dog) virtual medical assistant.' "
     "Use three sentences maximum and keep the answer concise.\n\n"),
    ("human", "{context}\n\n{input}"),
])

# Step 4: RAG pipeline
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

#  New: Friendly GET route for browser users
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to the Pet Medical Chatbot API!",
        # "usage": "Send a POST request to /chat with JSON: { 'message': 'your question' }"
    })

# POST /chat route
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message")

    if not user_input:
        return jsonify({"error": "Message field is required"}), 400

    response = rag_chain.invoke({"input": user_input})
    return jsonify({"response": response["answer"]})

# Run the server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
