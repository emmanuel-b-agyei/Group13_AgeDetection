# Group13_AgeDetection

# Age Detection through Face Recognition using Convolutional Neural Networks

## Overview

This project, "Age Detection through Face Recognition using Convolutional Neural Networks (CNN)," aims to accurately predict the age of an individual based on their facial features. The project utilizes computer vision techniques and deep learning, with a focus on Convolutional Neural Networks. The motivation behind the project lies in the growing importance of age prediction in various applications, including personalized user experiences, security systems, and age-specific content delivery.

## Features

- **Image Upload:** Users can upload an image, and the application will display the image along with the predicted age group.
- **Webcam Capture:** Users can use their webcam to capture real-time images, and the application will predict the age group for each captured frame.

## Project Structure

- The project involves dataset preprocessing, model development, training, and evaluation.
- The primary dataset used is the UTKFace.
- The CNN model is created using TensorFlow/Keras, and hyperparameter tuning is performed using GridSearchCV.
- The best model is evaluated on the test set, and the results include accuracy, a confusion matrix, and a classification report.

## Technical Requirements

- **Programming Language:** Python
- **Libraries:** TensorFlow/Keras, OpenCV, scikit-learn, Streamlit
- **Hardware:** A machine with GPU support for efficient model training
- **Development Environment:** Google Colab

## Setup

1. **Clone the repository:**

2. **Install the required dependencies:**

3. **Run the Python script for model training and evaluation.**

## Deployment

### Local Deployment

1. After model training, save the best model as 'best_model.h5'.
2. Run the Streamlit app locally: streamlit run streamlit_app.py
3. Access the application in your web browser at `http://localhost:8501`.

## Video Demonstration

For a detailed explanation and demonstration of the application, please watch our video on YouTube. Link: https://youtu.be/7J3Enp0X64E 

## Credits

This project was developed by Emmanuel Brewu Agyei and David Paa Kwesi Acquaah.
