import torch 
import cv2
from autocrop import autocrop


img_path = "sample/img-1.jpg" # support pillow image, numpy image and image path
output_path = "sample/result-img-1.png"
model_path = "models/autocrop_model_v2.pth" # support both .onnx and .pth

if torch.cuda.is_available():
    device = "cuda" # Use NVIDIA GPU (if available)
elif torch.backends.mps.is_available():
    device = "mps" # Use Apple Silicon GPU (if available)
else:
    device = "cpu" # Default to CPU if no GPU is available

# Call the autocrop function and store the result in a variable
cropped_image = autocrop(img_path=img_path, model_path=model_path, device=device)

# Save the result
cv2.imwrite(output_path, cropped_image[:, :, ::-1])  # Convert back to BGR for saving


print(f"Extracted document saved to {output_path}")


