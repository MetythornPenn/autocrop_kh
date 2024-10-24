# Autocrop_kh

#### Automatic Document Segmentation and Cropping for Khmer IDs, Passport and Documents

Autocrop_kh is a Python package for automatic document segmentation and cropping, with a focus on Khmer IDs, Passport and other documents. It uses a DeepLabV3 model training on Khmer ID, Passport document datasets to accurately segment and extract documents from images.

License: [Apache-2.0 License](https://github.com/MetythornPenn/sdab/blob/main/LICENSE)

## Installation

#### Install from source

```sh

# clone repo 
git clone https://github.com/MetythornPenn/autocrop_kh.git

# install lib from source
pip install -e .

```

#### Install from PyPI
```sh
pip install autocrop-kh
```

## Usage

#### Python Script

```python
import torch
import cv2
import requests
import os
from autocrop_kh import autocrop

# Function to download files from URLs
def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded: {local_filename}")

# URLs for the image and model
img_url = "https://github.com/MetythornPenn/autocrop_kh/raw/main/sample/img-1.jpg"
model_url = "https://github.com/MetythornPenn/autocrop_kh/raw/main/models/autocrop_model_v2.onnx"

# Local paths to save the files
img_path = "img-1.jpg"
model_path = "autocrop_model_v2.onnx"

# Download the image and model files
download_file(img_url, img_path)
download_file(model_url, model_path)

# Verify the files are correctly downloaded
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image file {img_path} was not found.")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} was not found.")

# Specify device (CPU or CUDA or Apple Silicon GPU)
if torch.cuda.is_available():
    device = "cuda"  # Use NVIDIA GPU (if available)
elif torch.backends.mps.is_available():
    device = "mps"  # Use Apple Silicon GPU (if available)
else:
    device = "cpu"  # Default to CPU if no GPU is available

# Perform document extraction
extracted_document = autocrop(img_path=img_path, model_path=model_path, device=device)

# Save the extracted document
output_path = "extracted_document.jpg"
cv2.imwrite(output_path, extracted_document[:, :, ::-1])  # Convert back to BGR for saving

print(f"Extracted document saved to {output_path}")


```

- `img_path`: Path of the input image file.
- `model_path`: Path to the pre-trained model (local path and support both .onnx and .pth).
- `device`: Specify `cpu` or `cuda` or `mps` (default is `gpu`).
- `output_path`: Path where the extracted document image will be saved.

#### Result:

<p align="center">
  <img src="sample/img-1.jpg" alt="Left Image" width="45%">
  <img src="sample/result-img-1.png" alt="Right Image" width="45%">
</p>

<p align="center">
  <img src="sample/img-5.png" alt="Left Image" width="45%">
  <img src="sample/result-img-5.png" alt="Right Image" width="45%">
</p>


#### Running as API & Web
```sh
# clone repo
git clone https://github.com/MetythornPenn/autocrop_kh.git

# go to directory
cd autocrop

# install libraries
pip3 install -r requirements.txt

# start server (http://localhost:5555/docs)
make server

# start client ((http://localhost:7860))
make client 

```
**Noted** : This model was trained with 25000 datasets include opensource data and my custom synthetic data.
## Reference 
- Inspired by [DeepLabV3](https://paperswithcode.com/method/deeplabv3)
- [Publish python package to PyPI](https://www.youtube.com/watch?v=90PWQEc--6k)