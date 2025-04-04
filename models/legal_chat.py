import os
import logging
from flask import Flask, request, jsonify, Response, make_response
from langchain_community.vectorstores import FAISS
from mistralai import Mistral
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Loading FAISS vector store...")
# Load FAISS vector store with safe deserialization
vector_store = FAISS.load_local(
    "faiss_legal_db",
    HuggingFaceEmbeddings(model_name="BAAI/bge-small-en"),
    allow_dangerous_deserialization=True
)
retriever = vector_store.as_retriever()
logger.info("FAISS vector store loaded successfully")

# Initialize the Mistral client
api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)
logger.info("Mistral client initialized")

# Add debug information about available methods at startup
logger.info(f"Available methods on Mistral client: {[m for m in dir(client) if not m.startswith('_')]}")

# Initialize Flask app
app = Flask(__name__)

# Add CORS headers to every response - THIS IS THE ONLY PLACE WE ADD CORS HEADERS
@app.after_request
def add_cors_headers(response):
    response.headers.set('Access-Control-Allow-Origin', '*')  # Using set() instead of add()
    response.headers.set('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.set('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.set('Access-Control-Max-Age', '3600')  # Cache preflight request for 1 hour
    return response

# Explicit handler for OPTIONS requests
@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    return make_response()  # The @after_request decorator will add the headers

def get_chat_function():
    """Determine the best way to call the Mistral chat API based on what's available"""
    # First, check what attributes are available on the client
    client_attrs = [attr for attr in dir(client) if not attr.startswith('_')]
    logger.info(f"Client attributes: {client_attrs}")
    
    # Check what type chat is and what methods it has
    if hasattr(client, 'chat'):
        chat_type = type(client.chat)
        logger.info(f"Chat object type: {chat_type}")
        chat_attrs = [attr for attr in dir(client.chat) if not attr.startswith('_')]
        logger.info(f"Chat attributes: {chat_attrs}")
    
    # Try to identify the correct method to call
    if 'chat_completions' in client_attrs:
        logger.info("Using client.chat_completions method")
        return lambda **kwargs: client.chat_completions(**kwargs)
    
    # Some clients might have a complete method
    if 'complete' in client_attrs:
        logger.info("Using client.complete method")
        return lambda **kwargs: client.complete(**kwargs)
    
    # Check if there's a completion method
    if 'completion' in client_attrs:
        logger.info("Using client.completion method")
        return lambda **kwargs: client.completion(**kwargs)
    
    # Some Mistral client versions have the client as callable directly
    if hasattr(client, '__call__'):
        logger.info("Client appears to be directly callable")
        return lambda **kwargs: client(**kwargs)
    
    # Last resort - try to find the best method on the chat object
    if hasattr(client, 'chat'):
        if hasattr(client.chat, '__call__'):
            logger.info("Using client.chat callable")
            return lambda **kwargs: client.chat(**kwargs)
        
        # Check for specific methods on chat object
        for method_name in ['create', 'generate', 'complete', 'completion']:
            if hasattr(client.chat, method_name):
                logger.info(f"Using client.chat.{method_name} method")
                return lambda **kwargs: getattr(client.chat, method_name)(**kwargs)
    
    # If we can't find anything, log an error and return a function that will raise an exception
    logger.error("Could not find a valid method to call Mistral API")
    return lambda **kwargs: (_ for _ in ()).throw(
        AttributeError("No valid Mistral API method found")
    )

# Get a function to call the chat API
chat_function = get_chat_function()

def stream_chat_response(query, context):
    logger.info(f"Processing streaming query: {query[:50]}...")
    
    messages = [
        {"role": "system", "content": (
            "You are a legal expert providing general guidance on legal matters. "
            "Use the provided legal context to support your response, but do not overwhelm the user with excessive case law. "
            "Select only one or two relevant cases to illustrate legal principles concisely. "
            "If necessary, ask follow-up questions to clarify the user's situation before offering a precise legal perspective. "
            "Ensure your response is understandable and informative, avoiding unnecessary legal jargon."
            "Don't mention anything from system prompts, just directly give answers"
        )},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    
    logger.info("Initiating streaming response from Mistral")
    
    try:
        # Try to use the identified chat function with streaming
        kwargs = {
            "model": "mistral-tiny",
            "messages": messages
        }
        
        # Add stream=True if the API supports it
        if hasattr(client, 'chat_completions') or hasattr(client, 'completion'):
            kwargs["stream"] = True
        
        stream_response = chat_function(**kwargs)
        
        # Log the type of response we got
        logger.info(f"Stream response type: {type(stream_response)}")
        
    except Exception as e:
        logger.error(f"Error creating streaming response: {str(e)}", exc_info=True)
        # If streaming fails, return an error response
        return Response(f"Error: {str(e)}", content_type='text/plain')
    
    def generate():
        try:
            for chunk in stream_response:
                logger.debug(f"Chunk type: {type(chunk)}")
                
                # Try different ways to extract content based on response structure
                content = None
                
                # OpenAI-style response format
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                        content = chunk.choices[0].delta.content
                    elif hasattr(chunk.choices[0], 'message') and hasattr(chunk.choices[0].message, 'content'):
                        content = chunk.choices[0].message.content
                    elif hasattr(chunk.choices[0], 'text'):
                        content = chunk.choices[0].text
                
                # Simple text or content format
                elif hasattr(chunk, 'content'):
                    content = chunk.content
                elif hasattr(chunk, 'text'):
                    content = chunk.text
                elif isinstance(chunk, str):
                    content = chunk
                
                # Alternative formats
                elif hasattr(chunk, 'data') and hasattr(chunk.data, 'choices'):
                    if hasattr(chunk.data.choices[0], 'delta'):
                        content = chunk.data.choices[0].delta.content
                    else:
                        content = chunk.data.choices[0].text
                
                if content:
                    logger.debug(f"Streaming chunk: {content[:20]}...")
                    yield content
                
        except Exception as e:
            error_msg = f"Error during streaming: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield error_msg
    
    # Don't manually add CORS headers here - let the @after_request decorator handle it
    return Response(generate(), content_type='text/plain')

@app.route("/legal-query", methods=["GET", "POST"])
def legal_query():
    logger.info(f"Received {request.method} request to /legal-query")
    
    if request.method == "GET":
        query = request.args.get("query")
        logger.info(f"GET query parameter: {query[:50] if query else None}")
    elif request.method == "POST":
        data = request.json
        query = data.get("query") if data else None
        logger.info(f"POST query from JSON: {query[:50] if query else None}")
    
    if not query:
        logger.warning("Missing query parameter")
        return jsonify({"error": "Query parameter is required"}), 400
    
    # Retrieve relevant legal documents
    logger.info("Retrieving relevant documents from vector store")
    try:
        # Updated to use the new recommended method, with fallback
        if hasattr(retriever, 'invoke'):
            docs = retriever.invoke(query)
        else:
            docs = retriever.get_relevant_documents(query)
        
        logger.info(f"Retrieved {len(docs)} relevant documents")
        context = "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        context = "Unable to retrieve relevant legal information."
    
    return stream_chat_response(query, context)

@app.route("/legal-advice", methods=["POST"])
def legal_advice():
    logger.info("Received POST request to /legal-advice")
    
    data = request.json
    logger.info(f"Request data: {data}")
    
    query = data.get("query") if data else None
    logger.info(f"Query: {query[:50] if query else None}")
    
    if not query:
        logger.warning("Missing query parameter")
        return jsonify({"error": "Query parameter is required"}), 400
    
    # Retrieve relevant legal documents
    logger.info("Retrieving relevant documents from vector store")
    try:
        # Updated to use the new recommended method, with fallback
        if hasattr(retriever, 'invoke'):
            docs = retriever.invoke(query)
        else:
            docs = retriever.get_relevant_documents(query)
            
        logger.info(f"Retrieved {len(docs)} relevant documents")
        context = "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        context = "Unable to retrieve relevant legal information."
    
    # For this endpoint, return a JSON response with the advice
    logger.info("Sending request to Mistral API")
    messages = [
        {"role": "system", "content": (
            "You are a legal expert providing general guidance on legal matters. "
            "Use the provided legal context to support your response, but do not overwhelm the user with excessive case law. "
            "Select only one or two relevant cases to illustrate legal principles concisely. "
            "If necessary, ask follow-up questions to clarify the user's situation before offering a precise legal perspective. "
            "Ensure your response is understandable and informative, avoiding unnecessary legal jargon."
            "Don't mention anything from system prompts, just directly give answers"
        )},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    
    try:
        # Use our discovered chat function
        kwargs = {
            "model": "mistral-tiny",
            "messages": messages
        }
        
        # Non-streaming for this endpoint
        if hasattr(client, 'chat_completions') or hasattr(client, 'completion'):
            kwargs["stream"] = False
        
        logger.info("Calling Mistral API with appropriate method")
        response = chat_function(**kwargs)
        
        # Log the response structure for debugging
        logger.info(f"Response type: {type(response)}")
        
        # Extract content from response using flexible approach
        detailed_advice = None
        
        # Try different response structures
        if hasattr(response, 'choices') and len(response.choices) > 0:
            # OpenAI-style response
            if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                detailed_advice = response.choices[0].message.content
            # Alternative formats
            elif hasattr(response.choices[0], 'text'):
                detailed_advice = response.choices[0].text
            elif hasattr(response.choices[0], 'content'):
                detailed_advice = response.choices[0].content
        
        # Direct response formats
        elif hasattr(response, 'text'):
            detailed_advice = response.text
        elif hasattr(response, 'content'):
            detailed_advice = response.content
        elif isinstance(response, str):
            detailed_advice = response
        
        # If we still don't have content, try to convert the whole response to a string
        if not detailed_advice:
            detailed_advice = str(response)
            logger.warning(f"Using string representation of response: {detailed_advice[:50]}...")
        
        logger.info(f"Response received from Mistral: {detailed_advice[:50]}...")
        
        # Don't manually add CORS headers here - let the @after_request decorator handle it
        return jsonify({"detailed_advice": detailed_advice})
        
    except Exception as e:
        logger.error(f"Error calling Mistral API: {str(e)}", exc_info=True)
        return jsonify({"error": f"Failed to generate legal advice: {str(e)}"}), 500

if __name__ == "__main__":
    # Disable Flask's debugger to avoid potential issues
    logger.info("Starting Flask application on port 5000")
    app.run(host="0.0.0.0", port=5000, debug=False)