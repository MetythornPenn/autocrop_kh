import io
import torch
import cv2
import numpy as np
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from autocrop import autocrop, load_autocrop_model

app = FastAPI()

# Load the model at startup
@app.on_event("startup")
async def load_model():
    global model, device, model_type
    # model_path = "models/best_model.pth"
    model_path = "models/best-model.onnx"
    
    # Check if model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model once at startup
    model, model_type = load_autocrop_model(model_path, device)
    
    # Only set to eval mode if it's a PyTorch model
    if model_type == 'torch':
        model.eval()  # Set the model to evaluation mode


def process_image(image_data: bytes):
    # Print length of image_data for debugging
    print(f"Length of image_data: {len(image_data)}")

    # Convert image data to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    
    # Debug: Check first few bytes of the image data
    print(f"First few bytes of image_data: {image_data[:10]}")

    # Attempt to decode the image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Check if image decoding was successful
    if img is None:
        raise ValueError("Failed to decode image. The input data may not be a valid image.")

    # Call autocrop function using the loaded model
    cropped_img = autocrop(np_image=img, model_path="models/best_model.pth", device=device)

    # Convert cropped image back to bytes for returning
    _, img_encoded = cv2.imencode('.jpg', cropped_img)
    return img_encoded.tobytes()



@app.post("/crop-image/")
async def crop_image(file: UploadFile = File(...)):
    image_data = await file.read()
    cropped_image = process_image(image_data)
    return StreamingResponse(io.BytesIO(cropped_image), media_type="image/jpeg")

