import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Photo Editor", layout="wide")

st.title("📸 Photo Editor using OpenCV + Streamlit")

# ---------------------------
# Load Image
# ---------------------------
def load_image(file):
    image = Image.open(file).convert("RGB")
    return np.array(image)

# ---------------------------
# Image Processing Functions
# ---------------------------
def adjust_brightness(image, value):
    return cv2.convertScaleAbs(image, beta=value)

def adjust_contrast(image, value):
    return cv2.convertScaleAbs(image, alpha=value)

def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_blur(image, ksize):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def warm_filter(image):
    increase = np.array([10, 20, 30])
    return cv2.add(image, increase)

def sharpen(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def portrait_blur(image):
    blurred = cv2.GaussianBlur(image, (25, 25), 0)
    h, w, _ = image.shape
    center = (w//2, h//2)

    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, center, min(h, w)//3, 255, -1)
    mask = cv2.GaussianBlur(mask, (51, 51), 0)

    result = np.where(mask[:, :, np.newaxis] == 255, image, blurred)
    return result

# Extra Features
def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 100, 200)

def cartoon(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)

    edges = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 9, 9
    )

    color = cv2.bilateralFilter(image, 9, 250, 250)

    return cv2.bitwise_and(color, color, mask=edges)

# ---------------------------
# UI
# ---------------------------
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = load_image(uploaded_file)

    st.subheader("Original Image")
    st.image(image, width=500)

    # Sidebar Controls
    st.sidebar.header("Controls")

    # Resize
    width = st.sidebar.slider("Width", 100, 1000, image.shape[1])
    height = st.sidebar.slider("Height", 100, 1000, image.shape[0])

    # Brightness & Contrast
    brightness = st.sidebar.slider("Brightness", -100, 100, 0)
    contrast = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0)

    # Filters
    st.sidebar.subheader("Filters")
    grayscale = st.sidebar.checkbox("Grayscale")
    blur = st.sidebar.checkbox("Blur")
    warm = st.sidebar.checkbox("Warm Filter")
    sharp = st.sidebar.checkbox("Sharpen")
    portrait = st.sidebar.checkbox("Portrait Blur")

    # Extra Features
    st.sidebar.subheader("Extra Features")
    edge = st.sidebar.checkbox("Edge Detection")
    cartoon_effect = st.sidebar.checkbox("Cartoon Effect")

    # ---------------------------
    # Processing Pipeline
    # ---------------------------
    edited = image.copy()

    # Resize
    edited = cv2.resize(edited, (width, height))

    # Brightness & Contrast
    edited = adjust_brightness(edited, brightness)
    edited = adjust_contrast(edited, contrast)

    # Apply Filters
    if grayscale:
        edited = to_grayscale(edited)

    if blur:
        edited = apply_blur(edited, 15)

    if warm:
        edited = warm_filter(edited)

    if sharp:
        edited = sharpen(edited)

    if portrait:
        edited = portrait_blur(edited)

    if edge:
        edited = edge_detection(edited)

    if cartoon_effect:
        edited = cartoon(edited)

    # ---------------------------
    # Display Edited Image
    # ---------------------------
    st.subheader("Edited Image")
    st.image(edited, width=500)

    # ---------------------------
    # Download Button (FIXED)
    # ---------------------------
    result = Image.fromarray(edited if len(edited.shape) == 3 else cv2.cvtColor(edited, cv2.COLOR_GRAY2RGB))

    buf = io.BytesIO()
    result.save(buf, format="PNG")

    st.download_button(
        label="Download Image",
        data=buf.getvalue(),
        file_name="edited_image.png",
        mime="image/png"
    )