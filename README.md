---
Experince the Fake Image Detection app using the Streamlit link:
https://fake-image-detection.streamlit.app/

# Fake Image Detection

## Problem Statement

The rise of AI-generated fake images poses significant challenges in various fields, including media integrity and security. These convincing fakes can spread misinformation and create mistrust. To address this issue, we have developed a machine learning-based app to detect the authenticity of images.


## Model and Techniques

### Model: MesoNet
We have employed the **MesoNet architecture** for our image authenticity detection task. MesoNet is designed to identify manipulated or fake images, particularly focusing on mesoscopic (mid-level) image features. This architecture is known for its efficiency and effectiveness in detecting deepfake images.

### Techniques Used
1. **Ensemble Learning:** We have utilized an ensemble of MesoNet models to improve the robustness and accuracy of our predictions. The ensemble approach combines predictions from multiple models to produce a final, more reliable output.
2. **Gaussian Blur:** A Gaussian blur is applied to the input images during preprocessing to help smooth out noise and reduce detail, aiding in the detection of manipulated regions in images.

## Preprocessing Techniques
1. **Resizing:** Images are resized to a standard dimension of 256x256 pixels to match the input size expected by the MesoNet models.
2. **Gaussian Blur:** Applied to the images to enhance the model's ability to detect anomalies by smoothing out unnecessary details.

## Code Explanation

### Streamlit Application
We have created a Streamlit application that allows users to upload an image and receive a prediction indicating whether the image is real or fake. The application uses an ensemble of pre-trained MesoNet models for prediction.

### Main Functions
1. **get_models:** Loads and compiles multiple MesoNet models for ensemble predictions.
2. **apply_gaussian_blur:** Applies Gaussian blur to an input image.
3. **preprocess_single_image:** Resizes and applies Gaussian blur to an input image.
4. **ensemble_predict_single_image:** Preprocesses the image and uses the ensemble models to predict its authenticity.
5. **classify_image:** Integrates with the Streamlit interface to classify the uploaded image and display the result.

### Streamlit Interface
The Streamlit interface allows users to upload an image file, which is then processed and classified as either real or fake based on the model's prediction.

## Conclusion
This project demonstrates the use of MesoNet and ensemble learning to effectively detect fake images. By applying preprocessing techniques like Gaussian blur and resizing, we enhance the model's ability to identify subtle manipulations. The Streamlit application provides an easy-to-use interface for users to upload images and get immediate feedback on their authenticity.

## How to Run
1. Ensure you have all dependencies installed by running:
   ```sh
   pip install -r requirements.txt
   ```
2. Start the Streamlit application with the command:
   ```sh
   streamlit run util.py
   ```

## Requirements
The `requirements.txt` file includes all necessary libraries:
```
streamlit
tensorflow
Pillow
numpy
```

## Contributors

- Chode Nithin/ @Chode-Nithin
- Chokkapu Monisha/ @chokkapumonisha
## Feedback and Support

For any issues or suggestions, please contact nithinchode@gmail.com. We appreciate your feedback!


---
