import os
import json
import numpy as np
import tensorflow as tf
from keras.models import load_model, Model
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
import cv2
from flask import Flask, request, render_template

# --- Initial Setup ---
app = Flask(__name__)

# Ensure required directories exist
os.makedirs('static/uploads', exist_ok=True)

# --- Load Model and Class Names ---
print("🔄 Loading trained model...")
MODEL_PATH = 'saved_model/skin_model.h5'
model = load_model(MODEL_PATH)

print("📂 Loading class names...")
CLASS_NAMES_PATH = 'saved_model/class_names.json'
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)
    class_names = {int(k): v for k, v in class_names.items()}

print("✅ Model and class names loaded successfully.")


# --- Utility Functions ---

def get_img_array(img_path, size):
    """Loads image and converts it into batch of tensors."""
    img = image.load_img(img_path, target_size=size)
    array = image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model):
    """
    Generates Grad-CAM heatmap by dynamically finding the last convolutional layer.
    """
    # Find the last convolutional layer in the entire model
    last_conv_layer = None
    for layer in reversed(model.layers):
        # Check both layer name and output shape for convolutional layers
        if 'conv' in layer.name.lower() and len(layer.output_shape) == 4:
            last_conv_layer = layer
            break
    
    if last_conv_layer is None:
        raise ValueError("Could not find a suitable convolutional layer in the model.")

    # Create model that maps input to last conv layer + output predictions
    grad_model = Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    # Calculate gradients
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Generate heatmap
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path="static/gradcam.jpg", alpha=0.4):
    """Superimposes heatmap onto original image and saves the result with proper color handling."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    
    # Resize and prepare heatmap
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert to RGB
    
    # Superimpose images
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    
    # Convert back to BGR for saving
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(cam_path, superimposed_img)


# --- Routes ---

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# ... (previous imports and setup) ...

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "Error: No file part"
    file = request.files['file']
    if file.filename == '':
        return "Error: No selected file"

    if file:
        basepath = os.path.dirname(__file__)
        uploads_folder = os.path.join(basepath, 'static', 'uploads')
        os.makedirs(uploads_folder, exist_ok=True)
        
        # Save original file
        file_path = os.path.join(uploads_folder, file.filename)
        file.save(file_path)

        # Process image
        img_array = get_img_array(file_path, size=(224, 224))
        img_array_preprocessed = preprocess_input(img_array)

        # Make prediction
        preds = model.predict(img_array_preprocessed)
        predicted_class_index = int(np.argmax(preds[0]))
        predicted_class_name = class_names[predicted_class_index]
        confidence = round(100 * np.max(preds[0]), 2)

        # Generate Grad-CAM heatmap with unique filename
        gradcam_filename = None
        try:
            heatmap = make_gradcam_heatmap(img_array_preprocessed, model)
            
            # Create unique filename based on original file
            original_name = os.path.splitext(file.filename)[0]
            gradcam_filename = f"{original_name}_heatmap.jpg"
            cam_path = os.path.join(uploads_folder, gradcam_filename)
            
            save_and_display_gradcam(file_path, heatmap, cam_path)
        except Exception as e:
            print(f"⚠️ Grad-CAM failed: {str(e)}")

        return render_template(
            'result.html',
            prediction=predicted_class_name,
            confidence=confidence,
            original_image_path=f'uploads/{file.filename}',
            gradcam_image_path=f'uploads/{gradcam_filename}' if gradcam_filename else None
        )

if __name__ == '__main__':
    app.run(debug=True)