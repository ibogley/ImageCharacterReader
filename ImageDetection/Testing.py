import torch
import torchvision
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import os
from CharacterClassification.Models.Model2.Training import Model2


# Load pre-trained Faster R-CNN model
detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
detector.eval()
detector = detector.cuda() if torch.cuda.is_available() else detector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model2(num_classes=94)
model.load_state_dict(torch.load('CharacterClassification/Models/Model2/model2.pth', map_location=device))
model.to(device)
model.eval()

# Preprocessing for detection
detection_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Preprocessing for your CNN classifier
classification_transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def detect_and_classify(image_path, detector, model, detection_threshold=0.7, classification_threshold=0.8):
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_tensor = detection_transform(image).unsqueeze(0).to(device)

    # Step 1 — Detect objects
    with torch.no_grad():
        detections = detector(image_tensor)[0]

    # Step 2 — Loop over detections
    results = []
    img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for idx, score in enumerate(detections['scores']):
        if score < detection_threshold:
            continue

        # Get bounding box coordinates
        box = detections['boxes'][idx].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box

        # Crop detected region
        cropped = image.crop((x1, y1, x2, y2))

        # Preprocess for your CNN
        cropped_tensor = classification_transform(cropped).unsqueeze(0).to(device)

        # Step 3 — Classify cropped object
        with torch.no_grad():
            outputs = model(cropped_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_prob, pred_class = torch.max(probs, 1)

        pred_prob = pred_prob.item()
        pred_class = pred_class.item()

        if pred_prob < classification_threshold:
            label = "Not a character"
        else:
            label = f"Character {pred_class}"

        results.append((label, pred_prob, (x1, y1, x2, y2)))

        # Draw results on image
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_cv2, f"{label} ({pred_prob:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save or display result
    cv2.imwrite(image_path.replace(".jpg", "_classified.jpg"), img_cv2)
    return results


results = detect_and_classify("ImageDetection/IMG_1966.jpg", detector, model)

for label, prob, bbox in results:
    print(f"Detected: {label} | Confidence: {prob:.2f} | Box: {bbox}")

