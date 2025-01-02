# app.py (Updated Flask Backend)

from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load the trained model
model_path = "./model/model.h5"
model = load_model(model_path)

# Constants for preprocessing
WIDTH = 150
HEIGHT = 150

# Define class labels in the same order as they were during training
class_labels = ['Aom', 'Earwax', 'Normal', 'Others']

# Function to preprocess a single image
def preprocess_image(image_path):
    # Use load_img for consistent preprocessing
    img = load_img(image_path, target_size=(WIDTH, HEIGHT))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the image file is uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']

    # Save the uploaded image temporarily
    temp_path = "./temp_image.tiff"
    file.save(temp_path)

    try:
        # Preprocess the image
        preprocessed_image = preprocess_image(temp_path)

        # Make prediction
        predictions = model.predict(preprocessed_image)
        predictions = predictions[0]  # Get the first (and only) batch

        # Get class probabilities and the top prediction
        class_probabilities = {class_labels[i]: float(predictions[i]) for i in range(len(class_labels))}
        predicted_index = np.argmax(predictions)
        predicted_class = class_labels[predicted_index]
        confidence_score = float(predictions[predicted_index])

        # Delete the temporary file
        os.remove(temp_path)

        return jsonify({
            "predicted_class": predicted_class,
            "confidence_score": confidence_score,
            "class_probabilities": class_probabilities
        })
    except Exception as e:
        # Handle errors gracefully
        os.remove(temp_path)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
