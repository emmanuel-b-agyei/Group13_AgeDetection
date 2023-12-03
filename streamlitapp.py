import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image

# Load the trained model
best_model = load_model('best_model_1.h5')

# Define the mapping of encoded age group to age group labels
age_group_mapping = {
    0: '1-2',
    1: '3-6',
    2: '7-12',
    3: '13-17',
    4: '18-25',
    5: '26-32',
    6: '33-39',
    7: '40-46',
    8: '47-52',
    9: '53-59',
    10: '60-66',
    11: '67-74',
    12: '75-82',
    13: '83-100'
}

def preprocess_image(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to match the model's expected sizing
    resized_image = cv2.resize(gray_image, (150, 150))

    # Convert the image to a numpy array, normalize, and return
    return img_to_array(resized_image) / 255.0




def predict_age(image):
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Reshape the image to (1, 150, 150, 1) to match the model input format
    reshaped_image = np.reshape(processed_image, (1, 150, 150, 1))

    # Make predictions
    predictions = best_model.predict(reshaped_image)

    # Get the predicted age group
    predicted_age_group_encoded = np.argmax(predictions)

    # Map the encoded age group to the original age group label
    predicted_age_group = age_group_mapping.get(predicted_age_group_encoded, 'Unknown')

    return predicted_age_group


def main():

    st.title("Age Prediction App")

    # Choose mode: "Upload" or "Webcam"
    mode = st.radio("Select Mode:", ["Upload", "Webcam"])

    if mode == "Upload":
        uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Convert PIL image to numpy array
            image_np = np.array(image)
            # Predict age
            predicted_age_group = predict_age(image_np)
            st.subheader(f"Predicted Age Group: {predicted_age_group}")

    elif mode == "Webcam":
        # Initialize OpenCV capture
        video_capture = cv2.VideoCapture(0)

        # Process only one frame from the webcam
        _, frame = video_capture.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption="Webcam Feed", use_column_width=True)

        # Predict the age
        predicted_age_group = predict_age(frame)
        st.subheader(f"Predicted Age Group: {predicted_age_group}")

        # Break out of the loop to capture only one frame
        st.stop()


if __name__ == '__main__':
    main()
