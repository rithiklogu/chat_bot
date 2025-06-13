from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ API Key. Check your .env file.")


app = Flask(__name__)

db_directory = "chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
docsearch = Chroma(persist_directory=db_directory, embedding_function=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 1})

llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.5, max_tokens=500)

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a highly reliable medical assistant for cats and dogs. "
     "Answer the question **only** using the provided context below. "
     "If the answer is not present in the context, clearly reply: "
     "'This is out of context, I am a helpful pet's (Cat and Dog) virtual medical assistant.' "
     "Do not guess or make up information. Keep the response factual, concise (max 3 sentences), and context-grounded."),
    
    ("human", "{context}\n\n{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

#  
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to the Pet Medical Chatbot API!",

    })


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message")

    if not user_input:
        return jsonify({"error": "Message field is required"}), 400

    response = rag_chain.invoke({"input": user_input})
    return jsonify({"response": response["answer"]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
