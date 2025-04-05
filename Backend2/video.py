from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import cv2
from PIL import Image
from collections import Counter
import torchvision.transforms as transforms
import os
from flask_cors import CORS
import time
from huggingface_hub import hf_hub_download

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Class mapping
class_mapping = {
    0: "Abuse",
    1: "Arrest",
    2: "Arson",
    3: "Assault",
    4: "Burglary",
    5: "Explosion",
    6: "Fighting",
    7: "Normal"
}

# Constants and transformer
RESOLUTION = 224
transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.Resize((RESOLUTION, RESOLUTION))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
repo_id = "namban4123/crimemodel"  
filename = "crime_tcn_jit.pt" 
scripted_model_path = hf_hub_download(repo_id=repo_id, filename=filename)
model = torch.jit.load(scripted_model_path, map_location=device)
model.to(device)
model.eval()

# Preprocessing function
def preprocess_frame(frame):
    frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    Y_channel, _, _ = cv2.split(frame_yuv)
    pil_frame = Image.fromarray(Y_channel)
    input_tensor = transformer(pil_frame)
    return input_tensor.unsqueeze(0)

# Inference function
def infer_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_predictions = []
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        input_tensor = preprocess_frame(frame)
        with torch.no_grad():
            output = model(input_tensor.to(device))
            predicted = output.argmax(dim=1).item()
            frame_predictions.append(predicted)
    
    cap.release()
    
    final_prediction = Counter(frame_predictions).most_common(1)[0][0]
    inference_time = time.time() - start_time
    
    return final_prediction, inference_time

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files['video']
    temp_video_path = "temp_video.mp4"
    video_file.save(temp_video_path)
    
    prediction, inference_time = infer_video(temp_video_path)
    os.remove(temp_video_path)

    # Map the prediction to the human-readable class name
    predicted_class_name = class_mapping.get(prediction, "Unknown")

    # Print the prediction and inference time
    print(f"Prediction: {predicted_class_name}, Inference Time: {inference_time:.2f} seconds")
    
    return jsonify({
        "predicted_class": predicted_class_name,
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
