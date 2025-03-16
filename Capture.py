import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore
import os

# Load YOLOv8 models
classification_model = YOLO(r"best (8).pt")  # Classification Model
segmentation_model = YOLO(r"best (9).pt")  # Segmentation Model

# Firebase Setup
firebase_key_path = r"firebase_key.json"
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_key_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()

CATEGORY_NAMES = {
    0: "Acne", 1: "Dark Circle", 2: "Dark Spot", 3: "Dry Skin",
    4: "Normal Skin", 5: "Oily Skin", 6: "Pores",
    7: "Skin Redness", 8: "Wrinkles"
}

CATEGORY_COLORS = {
    0: (0, 0, 255), 1: (128, 0, 128), 2: (0, 0, 128), 3: (165, 42, 42),
    4: (0, 255, 0), 5: (255, 165, 0), 6: (255, 255, 0),
    7: (255, 0, 0), 8: (192, 192, 192)
}

st.set_page_config(page_title="AI Skin Analysis", layout="wide")
st.title("Capture and Analyze Skin")

st.markdown("""
    <style>
        [data-testid="stSidebar"], [data-testid="collapsedControl"] {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

st.write("Click below to capture an image")
image_file = st.camera_input("Take a picture")

if image_file is not None:
    image = Image.open(image_file)
    image = np.array(image.convert('RGB'))

    # Convert image to BGR format for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # **Face Detection**
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

    if len(faces) == 0:
        st.error("No face detected. Please try again.")
    else:
        (x, y, w, h) = faces[0]
        face_only = image_bgr[y:y + h, x:x + w]
        image_path = "captured_face_only.jpg"
        cv2.imwrite(image_path, face_only)
        st.success(f"Face image saved as '{image_path}'")

        # **Skin Classification**
        results_classification = classification_model(image_path)
        skin_type = results_classification[0].names[results_classification[0].probs.top1]

        # **Skin Segmentation**
        results_segmentation = segmentation_model(image_path)
        mask = np.zeros_like(face_only, dtype=np.uint8)
        detected_conditions = set()

        for result in results_segmentation:
            if hasattr(result, "masks") and result.masks is not None:
                for seg_mask, cls in zip(result.masks.xy, result.boxes.cls):
                    points = np.array(seg_mask, np.int32)
                    cls_id = int(cls)
                    color = CATEGORY_COLORS.get(cls_id, (255, 255, 255))
                    cv2.fillPoly(mask, [points], color)
                    detected_conditions.add(cls_id)

        segmented_img = cv2.addWeighted(face_only, 0.7, mask, 0.3, 0)

        # **Summary Section**
        col1, col2 = st.columns([1, 1])

        with col1:
            resized_image = cv2.resize(face_only, (250, 200))
            st.subheader("Original Image")
            st.image(Image.fromarray(resized_image), caption="Captured Image", use_container_width=False)

        with col2:
            st.subheader("📝 Analysis Summary")
            skin_problems = [CATEGORY_NAMES[cls_id] for cls_id in detected_conditions if
                             CATEGORY_NAMES[cls_id] not in ["Normal Skin", "Oily Skin", "Dry Skin"]]

            col_skin_type, col_skin_problems = st.columns(2)

            with col_skin_type:
                st.markdown("### 🌿 Skin Type")
                st.markdown(f"- ✅ {skin_type}")

            with col_skin_problems:
                st.markdown("### ⚠️ Skin Problems")
                for problem in skin_problems:
                    st.markdown(f"- ❌ {problem}")
                if not skin_problems:
                    st.success("No major skin problems detected!")

        # **Product Recommendations with Arrows**
        for cls_id in detected_conditions:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader(f"Segmented Image for {CATEGORY_NAMES[cls_id]}")
                st.image(segmented_img, caption=f"{CATEGORY_NAMES[cls_id]}", channels="BGR")

            with col2:
                st.subheader(f"🛍️ Recommended Products for {CATEGORY_NAMES[cls_id]}")
                products_ref = db.collection('Products').where("Problem", "==", CATEGORY_NAMES[cls_id])
                products = list(products_ref.stream())

                if products:
                    product_index = st.session_state.get(f'product_index_{cls_id}', 0)
                    num_products = len(products)
                    num_cols = 3

                    cols_nav = st.columns([1, 3, 1])
                    with cols_nav[0]:
                        if st.button("⬅️", key=f'prev_{cls_id}'):
                            product_index = (product_index - num_cols) % num_products

                    with cols_nav[2]:
                        if st.button("➡️", key=f'next_{cls_id}'):
                            product_index = (product_index + num_cols) % num_products

                    st.session_state[f'product_index_{cls_id}'] = product_index

                    product_columns = st.columns(num_cols)
                    for i in range(num_cols):
                        product_idx = product_index + i
                        if product_idx < num_products:
                            product_data = products[product_idx].to_dict()
                            with product_columns[i]:
                                st.markdown(f"**{product_data.get('Product', 'Unnamed Product')}**")
                                st.image(product_data.get('Img_URL', ''), width=150)
                                st.write(product_data.get('Description', 'No description available.'))
                                st.write(f"💰 Price: ₹{product_data.get('Price', 'N/A')}")
                                st.markdown(f"[🛒 Buy Now]({product_data.get('Prod_URL', '#')})", unsafe_allow_html=True)

st.write("Press the camera button above to capture your image.")
