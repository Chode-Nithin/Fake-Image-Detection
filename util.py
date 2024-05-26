import tensorflow as tf
from PIL import Image, ImageFilter
import numpy as np
import streamlit as st

labels = ['real', 'fake']
probab = 0
st.title("Authenticity_Detection")
st.write("Upload the picture here!")

@st.cache_resource()
def get_models():
    # Load all models for ensemble
    model_paths = ['model11.h5', 'model2.h5', 'model3.h5']  # Replace with your actual model paths
    models = [tf.keras.models.load_model(model_path) for model_path in model_paths]
    return models

ensemble_models = get_models()

def apply_gaussian_blur(img_array):
    img = Image.fromarray(img_array)
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    return np.array(img)

def preprocess_single_image(image):
    # Resize the image to match the input size used during training
    img = image.resize((256, 256))
    # Convert image to numpy array
    img_array = np.array(img)
    # Apply Gaussian blur
    img_blurred = apply_gaussian_blur(img_array)
    return img_blurred

def ensemble_predict_single_image(image):
    # Preprocess the single image
    preprocessed_image = preprocess_single_image(image)
    
    predictions = []
    for ensemble_model in ensemble_models:
        # Reshape the preprocessed image to match the input shape expected by the model
        image_reshaped = np.expand_dims(preprocessed_image, axis=0)
        # Make prediction using the current model
        prediction = ensemble_model.predict(image_reshaped)
        predictions.append(prediction)
    # Calculate the mean prediction across all models
    ensemble_prediction = np.mean(predictions, axis=0)
    return ensemble_prediction

file_uploaded = st.file_uploader("Choose the Image File", type=["jpg", "png", "jpeg"])

def classify_image(file_uploaded):
    if file_uploaded is not None:
        image = Image.open(file_uploaded)  # Reading the image
        ensemble_prediction = ensemble_predict_single_image(image)
        
        # Determine the label based on the threshold
        if ensemble_prediction[0][0] > 0.32:
            label = 'fake'
        else:
            label = 'real'
        
        probab = float(ensemble_prediction[0][0])
        
        result = {
            'label': label,
            'probability': probab
        }
        
        return result
    else:
        return None

rs = classify_image(file_uploaded)

if rs is not None:
    st.write("Your Image is:", rs['label'])
    st.write("Probability:", rs['probability'])
