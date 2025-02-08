import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO

# Load the trained YOLO classification model
classification_model = YOLO(r"C:\Users\HP\Downloads\best (8).pt")  # Model for skin classification
segmentation_model = YOLO(r"C:\Users\HP\Downloads\best (9).pt")  # Model for skin segmentation

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the webcam
cap = cv2.VideoCapture(0)

# Define ellipse parameters for centering the face
ellipse_center = (int(cap.get(3)) // 2, int(cap.get(4)) // 2)
ellipse_axes = (85, 110)
size_factor = 1.2
padding_factor = 0.3

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))

    face_centered = False
    face_only = None
    original_frame = frame.copy()

    for (x, y, w, h) in faces:
        face_center = (x + w // 2, y + h // 2)

        if (ellipse_center[0] - 30 < face_center[0] < ellipse_center[0] + 30 and
                ellipse_center[1] - 30 < face_center[1] < ellipse_center[1] + 30):
            face_centered = True
            pad_w = int(w * padding_factor)
            pad_h = int(h * padding_factor)

            face_only = original_frame[y - pad_h:y + h + pad_h, x - pad_w:x + w + pad_w]

    # Draw the ellipse while capturing (only for display)
    color = (0, 0, 255) if not face_centered else (0, 255, 0)
    scaled_axes = (int(ellipse_axes[0] * size_factor), int(ellipse_axes[1] * size_factor))
    cv2.ellipse(frame, ellipse_center, scaled_axes, 0, 0, 360, color, 3)

    cv2.imshow('Face Detection - Press "S" to Save, "Q" to Quit', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and face_centered and face_only is not None:
        image_path = 'captured_face_only.jpg'
        cv2.imwrite(image_path, face_only)
        print(f"Face image saved as '{image_path}'")
        break
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# Wait for a moment to ensure the file is saved
time.sleep(1)

# **Step 2: Skin Classification**
results_classification = classification_model(image_path)

# Get the top classification result
skin_type = results_classification[0].names[results_classification[0].probs.top1]
print(f"Skin Type: {skin_type}")

# **Step 3: Skin Instance Segmentation**
CATEGORY_COLORS = {
    0: (0, 0, 255),      # Acne - Red
    1: (128, 0, 128),    # Dark Circle - Purple
    2: (0, 0, 128),      # Dark Spot - Dark Blue
    3: (165, 42, 42),    # Dry Skin - Brown
    4: (0, 255, 0),      # Normal Skin - Green
    5: (255, 165, 0),    # Oily Skin - Orange
    6: (255, 255, 0),    # Pores - Yellow
    7: (255, 0, 0),      # Skin Redness - Bright Red
    8: (192, 192, 192)   # Wrinkles - Silver/Grey
}

# Perform segmentation
results_segmentation = segmentation_model(image_path)

# Load the captured image
image = cv2.imread(image_path)

if image is None:
    print("Error: Unable to read the captured image.")
    exit()

# Create an empty mask for segmentation
mask = np.zeros_like(image, dtype=np.uint8)

for result in results_segmentation:
    if hasattr(result, "masks") and result.masks is not None:
        for seg_mask, cls in zip(result.masks.xy, result.boxes.cls):
            points = np.array(seg_mask, np.int32)
            color = CATEGORY_COLORS.get(int(cls), (255, 255, 255))
            cv2.fillPoly(mask, [points], color)

# Overlay the mask on the original image
segmented_img = cv2.addWeighted(image, 0.7, mask, 0.3, 0)

# Display the classified skin type on the image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(segmented_img, f"{skin_type}", (30, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

# Show the segmented result
cv2.imshow("Segmented Image", segmented_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
