"""
app.py

Streamlit demo: upload an image and run person detection with posture classification.
Uses custom trained YOLOv8 model for high-accuracy person detection.

Usage:
    streamlit run app.py
"""

import cv2
import streamlit as st
from PIL import Image
import numpy as np
import urllib.request
import os

st.set_page_config(page_title="Human-Posture-CV Demo", layout="centered")

st.title("Human-Posture-CV — Detection Demo")
st.caption(
    "Person detector with posture classification (crouching, standing, sleeping, sitting, walking, running). "
    "Upload an image to run inference."
)

conf = st.slider("Detection confidence", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_model():
    """
    Load custom trained YOLOv8 model for high-accuracy person detection.
    """
    onnx_path = "runs/train/exp/weights/best.onnx"
    
    if os.path.exists(onnx_path):
        try:
            net = cv2.dnn.readNetFromONNX(onnx_path)
            return net
        except Exception as e:
            st.warning(f"Custom model loading failed: {e}. Using HOG detector instead.")
            return None
    else:
        st.warning("Custom trained model not found. Using HOG detector instead.")
        return None

def classify_posture_from_bbox(x1, y1, x2, y2, img_height, img_width):
    """
    Classify posture based on bounding box geometry and position.
    Returns one of: crouching, standing, sleeping, sitting, walking, running
    """
    width = x2 - x1
    height = y2 - y1
    aspect_ratio = width / height
    
    # Calculate relative position in image
    center_y = (y1 + y2) / 2
    relative_y = center_y / img_height
    
    # Calculate if person is near bottom (sitting/crouching) or full height
    height_ratio = height / img_height
    
    # Check if person is horizontal (sleeping) - wide aspect ratio
    if aspect_ratio > 1.5:
        return "sleeping"
    
    # Check if person is crouching (low height, wide aspect ratio)
    if aspect_ratio > 0.8 and height_ratio < 0.4:
        return "crouching"
    
    # Check if person is sitting (moderate height, wide aspect ratio, lower in image)
    if aspect_ratio > 0.6 and height_ratio < 0.6 and relative_y > 0.4:
        return "sitting"
    
    # Check if person is running (taller aspect ratio, likely mid-stride)
    if aspect_ratio < 0.4 and height_ratio > 0.7:
        return "running"
    
    # Check if person is walking (similar to running but slightly different proportions)
    if aspect_ratio < 0.5 and height_ratio > 0.6:
        return "walking"
    
    # Default to standing
    return "standing"

def detect_persons_hog(img_bgr):
    """
    Fallback person detection using HOG with improved parameters for better detection.
    """
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Improved parameters for better detection:
    # - Lower scale for more detection steps
    # - Smaller winStride for more detailed scanning
    # - Higher padding to catch people at edges
    boxes, weights = hog.detectMultiScale(
        img_bgr, 
        winStride=(4, 4),      # Smaller stride for more detailed scanning
        padding=(8, 8),        # Padding to catch edge cases
        scale=1.02,            # Lower scale for more detection steps
        hitThreshold=0         # Lower threshold to detect more people
    )
    
    return boxes

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input image", use_container_width=True)

    with st.spinner("Running detection and posture classification..."):
        # Convert PIL image to numpy array
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_height, img_width = img_bgr.shape[:2]
        
        annotated = img_bgr.copy()
        posture_counts = {"crouching": 0, "standing": 0, "sleeping": 0, "sitting": 0, "walking": 0, "running": 0}
        
        # Try to load ONNX model, fallback to HOG
        net = load_model()
        
        if net is not None:
            # Use custom trained model for high-accuracy detection
            st.success("Using custom trained model for high-accuracy detection")
            
            # Preprocess image for YOLOv8
            input_size = 640
            blob = cv2.dnn.blobFromImage(img_bgr, 1/255.0, (input_size, input_size), swapRB=True, crop=False)
            net.setInput(blob)
            
            # Run inference
            outputs = net.forward()
            
            # YOLOv8 output processing
            # Output shape: (1, 84, 8400) where 84 = 4 (bbox) + 80 (classes)
            predictions = outputs[0]  # Remove batch dimension
            
            # Transpose to get (8400, 84)
            predictions = predictions.T
            
            # Filter for person class (class 0 in COCO) and confidence threshold
            person_class_idx = 4  # First 4 are bbox coordinates, class 0 starts at index 4
            boxes = []
            confidences = []
            
            for pred in predictions:
                # Get bbox coordinates (center_x, center_y, width, height)
                cx, cy, w, h = pred[:4]
                
                # Get person class confidence
                person_conf = pred[person_class_idx]
                
                if person_conf > conf:
                    # Convert from center format to corner format
                    x1 = cx - w/2
                    y1 = cy - h/2
                    x2 = cx + w/2
                    y2 = cy + h/2
                    
                    # Scale to original image size
                    x1 = x1 * img_width / input_size
                    y1 = y1 * img_height / input_size
                    x2 = x2 * img_width / input_size
                    y2 = y2 * img_height / input_size
                    
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(float(person_conf))
            
            # Apply NMS manually to avoid API compatibility issues
            if len(boxes) > 0:
                boxes = np.array(boxes)
                confidences = np.array(confidences)
                
                # Manual NMS implementation
                indices = []
                if len(boxes) > 0:
                    # Sort by confidence
                    sorted_indices = np.argsort(confidences)[::-1]
                    
                    while len(sorted_indices) > 0:
                        # Pick the box with highest confidence
                        current_idx = sorted_indices[0]
                        indices.append(current_idx)
                        
                        if len(sorted_indices) == 1:
                            break
                        
                        # Calculate IoU with remaining boxes
                        current_box = boxes[current_idx]
                        remaining_boxes = boxes[sorted_indices[1:]]
                        
                        # Calculate IoU
                        x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
                        y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
                        x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
                        y2 = np.minimum(current_box[3], remaining_boxes[:, 3])
                        
                        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
                        current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
                        remaining_areas = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
                        union = current_area + remaining_areas - intersection
                        
                        iou = intersection / (union + 1e-6)
                        
                        # Keep boxes with IoU below threshold
                        keep_mask = iou < 0.45
                        sorted_indices = sorted_indices[1:][keep_mask]
                
                for i in indices:
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    
                    # Classify posture
                    posture = classify_posture_from_bbox(x1, y1, x2, y2, img_height, img_width)
                    posture_counts[posture] = posture_counts.get(posture, 0) + 1
                    
                    # Draw bounding box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"{posture}", (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Fallback to HOG detector
            st.warning("Using HOG detector (DNN model unavailable)")
            boxes = detect_persons_hog(img_bgr)
            
            for (x, y, w, h) in boxes:
                # Classify posture
                posture = classify_posture_from_bbox(x, y, x + w, y + h, img_height, img_width)
                posture_counts[posture] = posture_counts.get(posture, 0) + 1
                
                # Draw bounding box
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(annotated, f"{posture}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert back to RGB for display
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    st.image(annotated_rgb, caption="Detections with posture", use_container_width=True)
    
    total_detections = sum(posture_counts.values())
    st.write(f"**{total_detections} person(s) detected**")
    
    # Show posture breakdown
    if total_detections > 0:
        st.subheader("Posture Breakdown:")
        for posture, count in posture_counts.items():
            if count > 0:
                st.write(f"- **{posture.capitalize()}**: {count}")
else:
    st.info("Upload an image to get started.")
