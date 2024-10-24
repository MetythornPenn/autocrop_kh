import os
import gc
import cv2
import numpy as np
import torch
import torchvision.transforms as torchvision_T
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
import onnxruntime as ort
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress ONNX Runtime warnings by setting environment variables
os.environ["ORT_LOGGING_LEVEL"] = "3"  # Set to 3 to suppress warnings and only show errors


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect.astype("int").tolist()

def find_dest(pts):
    (tl, tr, br, bl) = pts
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
    return order_points(destination_corners)

def image_preproces_transforms(mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109)):
    common_transforms = torchvision_T.Compose(
        [torchvision_T.ToTensor(), torchvision_T.Normalize(mean, std)]
    )
    return common_transforms

def load_autocrop_model(checkpoint_path, device):
    if checkpoint_path.endswith('.pth'):
        num_classes = 2
        model = deeplabv3_mobilenet_v3_large(num_classes=num_classes)
        model.to(device)
        checkpoints = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoints, strict=False)
        model.eval()
        return model, 'torch'
    elif checkpoint_path.endswith('.onnx'):
        # Load the ONNX model
        session = ort.InferenceSession(checkpoint_path, providers=['CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'])
        return session, 'onnx'
    else:
        raise ValueError("Unsupported model format. Supported formats are .pth and .onnx")

preprocess_transforms = image_preproces_transforms()

def extract(image_true=None, trained_model=None, image_size=384, BUFFER=10, device=None, model_type='torch'):
    IMAGE_SIZE = image_size
    half = IMAGE_SIZE // 2

    imH, imW, C = image_true.shape
    image_model = cv2.resize(image_true, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    scale_x = imW / IMAGE_SIZE
    scale_y = imH / IMAGE_SIZE

    image_model = preprocess_transforms(image_model)
    image_model = torch.unsqueeze(image_model, dim=0)

    if model_type == 'torch':
        image_model = image_model.to(device)
        with torch.no_grad():
            out = trained_model(image_model)["out"].cpu()
    elif model_type == 'onnx':
        image_model = image_model.numpy()  # Convert tensor to numpy for ONNX
        input_name = trained_model.get_inputs()[0].name
        output = trained_model.run(None, {input_name: image_model})
        out = torch.tensor(output[0])

    del image_model
    gc.collect()

    out = torch.argmax(out, dim=1, keepdims=True).permute(0, 2, 3, 1)[0].numpy().squeeze().astype(np.int32)
    r_H, r_W = out.shape

    _out_extended = np.zeros((IMAGE_SIZE + r_H, IMAGE_SIZE + r_W), dtype=out.dtype)
    _out_extended[half: half + IMAGE_SIZE, half: half + IMAGE_SIZE] = out * 255
    out = _out_extended.copy()

    del _out_extended
    gc.collect()

    canny = cv2.Canny(out.astype(np.uint8), 225, 255)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    page = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    epsilon = 0.02 * cv2.arcLength(page, True)
    corners = cv2.approxPolyDP(page, epsilon, True)
    corners = np.concatenate(corners).astype(np.float32)

    corners[:, 0] -= half
    corners[:, 1] -= half
    corners[:, 0] *= scale_x
    corners[:, 1] *= scale_y

    if not (np.all(corners.min(axis=0) >= (0, 0)) and np.all(corners.max(axis=0) <= (imW, imH))):
        left_pad, top_pad, right_pad, bottom_pad = 0, 0, 0, 0

        rect = cv2.minAreaRect(corners.reshape((-1, 1, 2)))
        box = cv2.boxPoints(rect)
        box_corners = np.int32(box)

        box_x_min = np.min(box_corners[:, 0])
        box_x_max = np.max(box_corners[:, 0])
        box_y_min = np.min(box_corners[:, 1])
        box_y_max = np.max(box_corners[:, 1])

        if box_x_min <= 0:
            left_pad = abs(box_x_min) + BUFFER

        if box_x_max >= imW:
            right_pad = (box_x_max - imW) + BUFFER

        if box_y_min <= 0:
            top_pad = abs(box_y_min) + BUFFER

        if box_y_max >= imH:
            bottom_pad = (box_y_max - imH) + BUFFER

        image_extended = np.zeros((top_pad + bottom_pad + imH, left_pad + right_pad + imW, C), dtype=image_true.dtype)
        image_extended[top_pad: top_pad + imH, left_pad: left_pad + imW, :] = image_true
        image_extended = image_extended.astype(np.float32)

        box_corners[:, 0] += left_pad
        box_corners[:, 1] += top_pad

        corners = box_corners
        image_true = image_extended

    corners = sorted(corners.tolist())
    corners = order_points(corners)
    destination_corners = find_dest(corners)
    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))

    final = cv2.warpPerspective(image_true, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv2.INTER_LANCZOS4)
    final = np.clip(final, a_min=0., a_max=255.)

    return final

def autocrop(img_path=None, np_image=None, pil_image=None, model_path=None, device=None):
    # Load the model and determine type (torch or onnx)
    trained_model, model_type = load_autocrop_model(checkpoint_path=model_path, device=device)

    if img_path:
        # If img_path is provided, read the image from disk
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)[:, :, ::-1]  # Convert BGR to RGB
    elif np_image is not None:
        # If np_image is provided, use it directly
        image = np_image  # Assuming np_image is already a NumPy array in RGB format
    elif pil_image is not None:
        # If pil_image is provided, convert it to NumPy array
        image = np.array(pil_image)  # Convert PIL image to NumPy array
    else:
        raise ValueError("No image input provided. Please provide img_path, np_image, or pil_image.")

    # Perform document extraction
    extracted_image = extract(image_true=image, trained_model=trained_model, device=device, model_type=model_type)

    return extracted_image

