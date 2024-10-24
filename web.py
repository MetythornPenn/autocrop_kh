import gradio as gr
import requests
from PIL import Image
from io import BytesIO

# Define the FastAPI server endpoint
API_URL = "http://127.0.0.1:5555/crop-image/"

def process_image(image):
    """Send the image to the FastAPI API and return both original and cropped image."""
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="JPEG")
    img_byte_arr = img_byte_arr.getvalue()

    # Send the image to the API
    files = {'file': img_byte_arr}
    response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        cropped_image = Image.open(BytesIO(response.content))
        return image, cropped_image
    else:
        return image, None

# Gradio interface
def inference(image):
    return process_image(image)

# Create Gradio UI with original and cropped image side-by-side
with gr.Blocks() as demo:
    gr.Markdown("## Image Cropper using Autocrop")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Upload an Image")
        
        with gr.Column():
            img_output = gr.Image(type="pil", label="Cropped Image")
    
    btn = gr.Button("Process Image")
    btn.click(inference, inputs=[img_input], outputs=[img_input, img_output])

# Run Gradio UI
if __name__ == "__main__":
    demo.launch()
