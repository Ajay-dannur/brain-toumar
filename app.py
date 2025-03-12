import os
os.system("pip install tensorflow gradio numpy pillow")  # Install required packages

import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("bone_xray_cnn_model.h5")

# Prediction function
def predict_bone_xray(img):
    img = img.resize((224, 224))  # Resize image
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = np.argmax(model.predict(img_array))

    if prediction == 0:
        return "Fractured Bone"
    else:
        return "Healthy Bone"

# Create Gradio interface
interface = gr.Interface(
    fn=predict_bone_xray,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Bone X-ray Classifier",
    description="Upload an X-ray image to detect bone fractures."
)

interface.launch()
