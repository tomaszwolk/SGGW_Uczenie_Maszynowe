import torch
import torchvision
from PIL import Image
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Images in the Image folder from: https://www.kaggle.com/datasets/kkhandekar/object-detection-sample-images?select=7.jpg

# Load a pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Function to load an image and convert it to a tensor
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = F.to_tensor(image)
    return image

# Function to perform object detection
def detect_objects(model, image, threshold=0.5):
    # Move the image to the same device as the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image = image.to(device)
    # Perform inference
    with torch.no_grad():
        outputs = model([image])
    # Filter out detections with a score below the threshold
    detections = outputs[0]
    scores = detections['scores']
    keep = scores >= threshold
    filtered_detections = {k: v[keep].cpu() for k, v in detections.items()}

    return filtered_detections

# Function to plot the image with detected bounding boxes
def plot_detections(image, detections):
    # Convert the tensor image to a numpy array and transpose it to [H, W, C] format
    image = image.permute(1, 2, 0).numpy()
    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 9))
    # Display the image
    ax.imshow(image)
    # Plot each bounding box
    for box in detections['boxes']:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle(
            (x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    # Set plot title and show plot
    ax.set_title('Object Detections')
    plt.show()


# Load an image
Tk().withdraw()
image_path = askopenfilename()
image = load_image(image_path)

# Perform object detection
detections = detect_objects(model, image, threshold=0.5)
# Plot the image with detections
plot_detections(image, detections)