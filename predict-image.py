import os
import json
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
import argparse

# --- Main Prediction Function ---
def predict_single_image(model, class_names_map, image_path):
    """
    Loads an image, preprocesses it, and returns the model's prediction.
    """
    print(f"📄 Loading image from: {image_path}")
    try:
        # Load and prepare the image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocess the image using the official MobileNetV2 function
        img_array_preprocessed = preprocess_input(img_array)

        print("🧠 Making prediction...")
        # Get model predictions
        preds = model.predict(img_array_preprocessed)
        
        # Decode the prediction
        predicted_class_index = np.argmax(preds[0])
        predicted_class_name = class_names_map.get(predicted_class_index, "Unknown Class")
        confidence = round(100 * np.max(preds[0]), 2)

        print("\n--- Prediction Result ---")
        print(f"🩺 Predicted Condition: {predicted_class_name}")
        print(f"🎯 Confidence: {confidence}%")
        print("-------------------------\n")

    except FileNotFoundError:
        print(f"❌ ERROR: The file was not found at '{image_path}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- Script Execution ---
if __name__ == '__main__':
    # Set up an argument parser to accept the image path from the command line
    parser = argparse.ArgumentParser(description="Classify a skin condition from an image.")
    parser.add_argument("image_path", type=str, help="The full path to the image file.")
    args = parser.parse_args()

    # --- Load Model and Class Names ---
    print("🔄 Loading trained model and class names...")
    try:
        MODEL_PATH = 'saved_model/skin_model.h5'
        model = load_model(MODEL_PATH)

        CLASS_NAMES_PATH = 'saved_model/class_names.json'
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
            # Convert string keys back to integer keys
            class_names = {int(k): v for k, v in class_names.items()}
        
        print("✅ Model and class names loaded successfully.")
        
        # Run the prediction
        predict_single_image(model, class_names, args.image_path)

    except FileNotFoundError:
        print("❌ ERROR: Could not find 'saved_model/skin_model.h5' or 'saved_model/class_names.json'.")
        print("Please make sure you have trained the model and these files exist.")
    except Exception as e:
        print(f"An error occurred during model loading: {e}")

