# 🐶🐱 Virtual Pet Medical Chatbot API (RAG + Groq + LLaMA3)

A RESTful chatbot service built using **Flask** and powered by **Groq’s LLaMA3-8B model**, enhanced with **RAG (Retrieval-Augmented Generation)** using a custom medical knowledge base for cats and dogs.

```bash
git clone https://github.com/yourusername/chat_bot.git
cd chat_bot

conda create -n rag_chat_bot python=3.10 -y
conda activate rag_chat_bot

pip install -r requirements.txt

GROQ_API_KEY=your_groq_api_key_here

run: python store_index.py

run: python app.py

It will run on * Running on http://127.0.0.1:8000 *

Test using Postman:
Method: GET
URL: http://127.0.0.1:8000 
You can see: Welcome to the Pet Medical Chatbot API!


Method: POST

URL: http://127.0.0.1:8000/chat

Body: raw → JSON

{
  "message": "What medicine is safe for cats with fever?"
}