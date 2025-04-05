#CRime LIve DEtection
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import io
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load pre-trained model
def load_model():
    # Load a pre-trained ResNet50 model
    model = resnet50(pretrained=True)
    
    # Modify the final layer for binary classification (crime or normal)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2)  # 2 classes: crime, normal
    
    # In a real scenario, you would load weights from your fine-tuned model
    # model.load_state_dict(torch.load('path_to_your_model_weights.pth'))
    
    model.eval()  # Set to evaluation mode
    return model

# Initialize model
model = load_model()

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define activity classes
classes = ['normal', 'crime']

# Simulated crime detection algorithm 
# In a real application, you would use your trained model's predictions
def detect_crime(frame):
    # Convert frame to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Apply transformations
    input_tensor = transform(pil_image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    
    # In a real scenario, you would use GPU if available
    # input_batch = input_batch.to('cuda')
    
    with torch.no_grad():
        start_time = time.time()
        output = model(input_batch)
        inference_time = time.time() - start_time
    
    # Get predictions
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # For demonstration, we'll use a simple heuristic to detect unusual activities
    # In a real application, this would be based on your model's actual predictions
    
    # Get top predictions
    confidence_normal = probabilities[0].item()
    confidence_crime = probabilities[1].item()
    
    # For demo purposes, use some visual features to "detect" crimes
    # This is just a placeholder for the actual model logic
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray, (21, 21), 0)
    
    # Calculate motion and brightness
    brightness = np.mean(frame_gray)
    
    # For demonstration, detect "unusual activity" based on brightness
    # In reality, you would use your trained model's predictions
    predictions = []
    
    # Simulate model prediction based on simple image properties
    # (In production, you would use the actual model output)
    if brightness < 100:  # Dark scenes might be suspicious in some contexts
        predictions.append({
            "prediction": "crime",
            "confidence": max(0.5, confidence_crime)
        })
    else:
        predictions.append({
            "prediction": "normal",
            "confidence": max(0.7, confidence_normal)
        })
    
    return predictions, inference_time

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the request
        data = request.json
        image_data = data['image'].split(',')[1]  # Remove the data:image/jpeg;base64, prefix
        
        # Decode the base64 image
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Detect crime in the frame
        predictions, inference_time = detect_crime(frame)
        
        return jsonify({
            'predictions': predictions,
            'inference_time': inference_time
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Crime Detection API server...")
    app.run(debug=True)
