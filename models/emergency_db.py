import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load JSON dataset
with open("models/fir_dataset_500.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Extract relevant text (combine subject and description for context)
documents = []
for entry in data:
    fir_text = f"Subject: {entry['subject']}\nDescription: {entry['description']}"
    documents.append(fir_text)

# Split text into chunks for embedding
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
document_chunks = text_splitter.create_documents(documents)

# Use BGE embeddings from HuggingFace
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

# Store in FAISS for fast retrieval
vector_store = FAISS.from_documents(document_chunks, embeddings)
vector_store.save_local("faiss_fir_db")

print("FIR dataset successfully stored in FAISS!")
