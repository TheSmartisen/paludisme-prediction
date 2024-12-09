from flask import Blueprint, request, jsonify
from config import Config
import tensorflow as tf
from app.modules.ia_module import load_and_predict_with_preprocessing

# Define the blueprint for the API
api_bp = Blueprint('api_bp', __name__)

@api_bp.route('/api/v1/health', methods=['GET'])
def health():
    return '', 200

@api_bp.route('/api/v1/predict-malaria', methods=['POST'])
def predict_malaria():
    try:
        # Vérifier si un fichier image est dans la requête
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        # Récupérer l'image du formulaire
        image_file = request.files['image']

        # Charger l'image en tant que tableau NumPy
        image = tf.io.decode_image(image_file.read(), channels=3)

        predictions, predictions_binary, percentage_scores = load_and_predict_with_preprocessing(
            Config.MODEL_FILE,
            Config.METADATA_FILE,
            [image]
        )

        # Convertir les résultats en types sérialisables
        predictions = predictions.squeeze().tolist()  # Convertir ndarray en liste
        percentage_scores = [float(score) for score in percentage_scores][0]  # Convertir les scores en float


        # Renvoyer la réponse avec la prédiction
        return jsonify({
            "prediction_probabilities": predictions,
            "prediction_score": percentage_scores,
            "prediction_binary": int(predictions_binary),
            "prediction_label": Config.LABEL_CLASS[int(predictions_binary)]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500