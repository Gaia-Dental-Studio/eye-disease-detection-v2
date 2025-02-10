# backend.py
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.image import resize
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load models
# For this multiclass model, please donwload on https://drive.google.com/file/d/1haaczI0ExuMDDrqeXi69rkZABhrM31cz/view?usp=drive_link
model_multiclass = tf.keras.models.load_model('models/final_model_20250206-040328_multiclass.h5', compile=False)
model_multiclass.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Individual disease models (modify paths to each model as required)
MODELS = {
    'cataract': tf.keras.models.load_model('models/final_model_20241031-091328_cataract_effnet.h5', compile=False),
    'diabetes': tf.keras.models.load_model('models/final_model_20241029-154908_diabetes_effnet.h5', compile=False),
    'glaucoma': tf.keras.models.load_model('models/final_model_20241031-085230_glaucoma_effnet.h5', compile=False),
    'hypertension': tf.keras.models.load_model('models/final_model_20241031-080449_hypertension_effnet.h5', compile=False),
    'other': tf.keras.models.load_model('models/final_model_20241031-093243_others_effnet.h5', compile=False)
}
for model in MODELS.values():
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Define prediction functions
def preprocess_image(image):
    """Preprocess input image to match model training pipeline."""
    image = image.resize((224, 224))  # Ensure correct input size
    image = np.array(image)
    image = effnet_preprocess(image)  # Use EfficientNet's preprocessing
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def load_and_preprocess_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB") # Handle various image formats and ensure RGB
    image = np.array(image)
    image = tf.convert_to_tensor(image) # Convert PIL Image to Tensorflow tensor
    image = resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image

def predict_multiclass(model, image):
    image = load_and_preprocess_image(image)
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=-1)
    confidence_scores = predictions[0] * 100  # Convert to percentage
    return predicted_class, confidence_scores


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

    # image = Image.open(file)
    if model_type == "Full Model":
        image_bytes = file.read()
        # predictions = predict_multiclass(image)
        MULTILABEL_CLASS = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']

        predicted_class, confidence_scores = predict_multiclass(model_multiclass, image_bytes)

        # Sort confidence scores and get top 3 in backend
        sorted_scores = sorted(zip(MULTILABEL_CLASS, confidence_scores), key=lambda item: item[1], reverse=True) #Zip for sorting
        top_3_scores = sorted_scores[:3]
        return jsonify({
            'best_prediction': MULTILABEL_CLASS[predicted_class[0]],
            'top_3_scores': [{'label': label, 'score': float(score)} for label, score in top_3_scores] #Format for JSON
        })

    else:
        image = Image.open(file)
        results = predict_partial(image)
        results = {key: float(value) if isinstance(value, np.float32) else value for key, value in results.items()}

        formatted_results = {disease: round(confidence * 100, 2) for disease, confidence in results.items()}
        sorted_results = sorted(formatted_results.items(), key=lambda item: item[1], reverse=True)
        top_3_results = dict(sorted_results[:3])

        best_disease = sorted_results[0][0]  # Highest confidence disease

        return jsonify({
            "top_3_results": top_3_results,
            "best_disease": best_disease,
            "all_results": formatted_results # All results for detailed view if needed
        })

if __name__ == '__main__':
    app.run(debug=True)