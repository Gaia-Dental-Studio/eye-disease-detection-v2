# backend.py
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib

app = Flask(__name__)

# Load models
model_multiclass = tf.keras.models.load_model('models/final_model_20240122-033648_multiclass.h5', compile=False)
model_multiclass.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Individual disease models (modify paths to each model as required)
MODELS = {
    'cataract': tf.keras.models.load_model('models/final_model_20240119-154434_cataract_effnet.h5', compile=False),
    'diabetes': tf.keras.models.load_model('models/final_model_20240119-172037_diabetes_effnet.h5', compile=False),
    'glaucoma': tf.keras.models.load_model('models/final_model_20240119-174234_glaucoma_effnet.h5', compile=False),
    'hypertension': tf.keras.models.load_model('models/final_model_20240122-074201_hypertension_effnet.h5', compile=False),
    'other': tf.keras.models.load_model('models/final_model_20240122-122757_others_effnet.h5', compile=False)
}
for model in MODELS.values():
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Define prediction functions
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_multiclass(image):
    processed_image = preprocess_image(image)
    predictions = model_multiclass.predict(processed_image)
    return predictions.tolist()  # Convert to list for JSON serialization

def predict_partial(image):
    processed_image = preprocess_image(image)
    results = {}
    for disease, model in MODELS.items():
        prediction = model.predict(processed_image)
        results[disease] = prediction[0][0]
    return results

# Flask route for the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    model_type = request.form.get('model_type')

    if not file or model_type not in ["Full Model", "Partial Model"]:
        return jsonify({"error": "Invalid input"}), 400

    image = Image.open(file)
    if model_type == "Full Model":
        predictions = predict_multiclass(image)
        return jsonify({"predictions": predictions})
    else:
        results = predict_partial(image)
        return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)