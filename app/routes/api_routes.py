from flask import Blueprint, request, jsonify
from config import Config
import tensorflow as tf
from app.modules.ia_module import load_and_predict_with_preprocessing
import os
from werkzeug.utils import secure_filename
import base64

# Define the blueprint for the API
api_bp = Blueprint('api_bp', __name__)

for category in Config.LABEL_CLASS:
    for feedback_type in Config.FEEDBACK_TYPES:
        path = os.path.join(Config.BASE_DATA_PATH, category, feedback_type)
        os.makedirs(path, exist_ok=True)


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
    
@api_bp.route('/api/v1/feedback', methods=['POST'])
def feedback():
    try:
        data = request.get_json()

        label = data.get('label')
        is_correct = data.get('correct')
        base64_image = data.get('image')

        # Vérifier les entrées
        if label not in Config.LABEL_CLASS or is_correct not in [True, False]:
            return jsonify({"error": "Paramètres invalides"}), 400

        # Décoder l'image
        # Décoder l'image
        # Découper pour ne garder que les données après "base64,"
        header, base64_data = base64_image.split(",", 1)

        # Décoder et sauvegarder comme fichier image
        image_data = base64.b64decode(base64_data)

        # Déterminer le chemin de sauvegarde
        feedback_type = "Correct" if is_correct else "Incorrect"
        save_path = os.path.join(Config.BASE_DATA_PATH, label, feedback_type)

        # Générer un nom de fichier unique
        filename = secure_filename(f"feedback_{label}_{feedback_type}_{len(os.listdir(save_path)) + 1}.png")
        file_path = os.path.join(save_path, filename)

        # Sauvegarder l'image
        with open(file_path, "wb") as image_file:
            image_file.write(image_data)

        return jsonify({"message": f"Feedback enregistré avec succès."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500