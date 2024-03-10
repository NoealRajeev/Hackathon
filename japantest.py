import os
import json
from xml.etree import ElementTree as ET
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from bs4 import BeautifulSoup
import torch

# Load pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()


# Function to parse XML annotations and convert to COCO JSON format
def xml_to_coco(xml_dir):
    coco_data = {"images": [], "annotations": []}
    image_id = 0
    annotation_id = 0

    for filename in os.listdir(xml_dir):
        if filename.endswith(".xml"):
            xml_path = os.path.join(xml_dir, filename)
            with open(xml_path, "r") as f:
                xml_content = f.read()
            soup = BeautifulSoup(xml_content, "xml")

            # Add image data to COCO JSON
            image_filename = soup.find("filename").text
            image_data = {
                "id": image_id,
                "file_name": image_filename,
                "height": int(soup.find("size").find("height").text),
                "width": int(soup.find("size").find("width").text)
            }
            coco_data["images"].append(image_data)

            # Add annotations to COCO JSON
            for obj in soup.find_all("object"):
                bbox = obj.find("bndbox")
                bbox_coords = [int(bbox.find(coord).text) for coord in ["xmin", "ymin", "xmax", "ymax"]]
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,  # Assuming there's only one class (road)
                    "bbox": bbox_coords,
                    "area": (bbox_coords[2] - bbox_coords[0]) * (bbox_coords[3] - bbox_coords[1]),
                    "iscrowd": 0
                }
                coco_data["annotations"].append(annotation)
                annotation_id += 1

            image_id += 1

    return coco_data


# Function to load and preprocess input image
def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image)
    return image_tensor


# Function to perform object detection
def detect_objects(image_path):
    # Load and preprocess input image
    image_tensor = load_and_preprocess_image(image_path)

    # Perform object detection
    with torch.no_grad():
        predictions = model([image_tensor])

    # Process predictions
    detected_objects = []
    for pred in predictions[0]["boxes"]:
        bbox = pred.cpu().numpy().astype(int)
        detected_objects.append(bbox)

    return detected_objects


# Function to draw bounding boxes on the image
def draw_boxes_on_image(image_path, detected_objects):
    # Load the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Draw bounding boxes on the image
    for bbox in detected_objects:
        draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline="red")

    # Display the image with bounding boxes
    image.show()


# Image and annotation directories
images_dir = r"C:\Users\rupal\Desktop\Japan\test\images"
annotations_dir = r"C:\Users\rupal\Desktop\Japan\train\annotations\xmls"

# Convert XML annotations to COCO JSON format
coco_data = xml_to_coco(annotations_dir)

# Save COCO JSON to a file
with open("annotations.json", "w") as f:
    json.dump(coco_data, f)

# Perform object detection on test images and draw bounding boxes
for filename in os.listdir(images_dir):
    if filename.endswith(".jpg"):
        image_path = os.path.join(images_dir, filename)
        detected_objects = detect_objects(image_path)
        print("Detected objects:", detected_objects)
        draw_boxes_on_image(image_path, detected_objects)
