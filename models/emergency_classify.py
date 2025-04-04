import os
from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from mistralai import Mistral
from mistralai.models import UserMessage

# Load FAISS vector store
vector_store = FAISS.load_local(
    "models/faiss_legal_db",
    HuggingFaceEmbeddings(model_name="BAAI/bge-small-en"),
    allow_dangerous_deserialization=True
)
retriever = vector_store.as_retriever()

# Initialize Mistral API client
api_key = os.environ.get("MISTRAL_API_KEY","HtNsZNzpfzLQyzlGTvkCwccUbASZaZNt")
client = Mistral(api_key=api_key)

# Flask app
app = Flask(__name__)

def classify_emergency(subject, description, context):
    """Uses Mistral to classify the FIR emergency level (1-5)."""
    
    messages = [
        {"role": "system", "content": (
            "You are an AI that classifies emergency levels (1-5) for FIR reports. "
            "Use past cases for reference, but focus on the severity of the crime. "
            "Return only a single number (1, 2, 3, 4, or 5) based on urgency. "
            "1 = least urgent (e.g., theft, minor disputes), "
            "5 = most urgent (e.g., murder, terrorism, violent assault)."
        )},
        {"role": "user", "content": f"Context:\n{context}\n\nSubject: {subject}\nDescription: {description}\n\nEmergency level:"}
    ]
    
    response = client.chat.complete(model="mistral-tiny", messages=messages)
    return response.choices[0].message.content

@app.route("/classify-fir", methods=["POST"])
def classify_fir():
    """Classifies the emergency level of an FIR report."""
    
    data = request.json
    subject = data.get("subject")
    description = data.get("description")

    if not subject or not description:
        return jsonify({"error": "Both 'subject' and 'description' are required"}), 400

    # Retrieve relevant past FIRs
    query = f"{subject} {description}"
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Get emergency level
    emergency_level = classify_emergency(subject, description, context)
    
    return jsonify({"subject": subject, "description": description, "emergency_level": emergency_level})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
