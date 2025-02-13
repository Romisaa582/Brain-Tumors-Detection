
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the model
model = load_model("brain_mri_model.keras")

# Classification labels
class_labels = {
    "Glioma": "Glioma Tumor",
    "Meningioma": "Meningioma Tumor",
    "No Tumor": "No Tumor Detected",
    "Pituitary": "Pituitary Tumor"
}

# Function to preprocess and predict
def predict(img):
    img = img.convert("RGB")
    img = img.resize((299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  
    prediction = model.predict(img_array)
    result_index = np.argmax(prediction)
    result_label = list(class_labels.keys())[result_index]
    confidence = np.max(prediction) * 100  
    return result_label, confidence

# Set Streamlit page configuration
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")

# Custom CSS for hiding file size
st.markdown("""
    <style>
    body, .stApp { background-color: #E3F2FD !important; }
    
    .title { font-size: 28px; font-weight: bold; color: #1565C0; text-align: center; }
    
    .result-box { 
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        font-size: 16px;
        width: 60%;
        margin: auto;
        border-left: 6px solid;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .positive { background-color: #F48FB1 !important; border-left-color: #AD1457 !important; }
    .negative { background-color: #C8E6C9 !important; border-left-color: #2E7D32 !important; }
    
    .upload-box, .footer, h3, p, label, .stCheckbox label, .stButton button, .image-caption {
        color: #1565C0 !important;
    }
    
    .footer {
        text-align: center;
        font-size: 12px;
        margin-top: 30px;
    }

    /* إخفاء حجم الملف */
    .uploadedFileInfo { display: none !important; }

    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown("<div class='title'>🧠 Brain Tumor Detection</div>", unsafe_allow_html=True)
st.markdown("---")

# Upload section
st.markdown("<div class='upload-box'>📤 Upload an MRI scan</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image_uploaded = Image.open(uploaded_file)
    
    show_image = st.checkbox("📷 Show Uploaded Image")
    if show_image:
        st.markdown("<p class='image-caption' style='text-align: center;'>Uploaded MRI Scan</p>", unsafe_allow_html=True)
        st.image(image_uploaded, use_column_width=True)
        st.markdown("---")

    st.write("🕵‍♂ Analyzing...")

    # Get prediction
    label, confidence = predict(image_uploaded)
    
    # Generate diagnosis message
    if label == "No Tumor":
        diagnosis = "✅ No Tumor Detected"
        box_class = "negative"
    else:
        diagnosis = f"⚠️ Tumor Detected: {class_labels[label]}"
        box_class = "positive"

    # Display result
    st.markdown(
        f"""
        <div class='result-box {box_class}'>
            <h3>{diagnosis}</h3>
            <p>🔬 Confidence: {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True
    )

# Footer
st.markdown("<div class='footer'>Developed with ❤️ using Streamlit</div>", unsafe_allow_html=True)
