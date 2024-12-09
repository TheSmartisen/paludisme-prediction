import tensorflow as tf
import numpy as np

def augment_image(image):
    image = tf.image.random_flip_left_right(image)  # Retournement horizontal
    image = tf.image.random_flip_up_down(image)    # Retournement vertical
    image = tf.image.random_brightness(image, max_delta=0.1)  # Changer la luminosité
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # Changer le contraste
    image = tf.image.rot90(image)  # Rotation
    return image

def load_and_predict_with_preprocessing(model_path, metadata_path, sample_images):
    # Charger le modèle
    loaded_model = tf.keras.models.load_model(model_path)
    
    # Charger le seuil et la taille des images depuis le fichier metadata
    with open(metadata_path, "r") as f:
        metadata = f.readlines()
    threshold = float(metadata[0].split(":")[1].strip())
    img_size = tuple(map(int, metadata[1].split(":")[1].strip().strip("()").split(",")))

    # Appliquer la data augmentation et redimensionner les images
    resized_images = [tf.image.resize(image, img_size).numpy() for image in sample_images]
    normalized_images = np.array(resized_images).astype("float32") / 255  # Normaliser entre 0 et 1
    augmented_images = [augment_image(tf.convert_to_tensor(image)) for image in normalized_images]

    # Convertir la liste des images augmentées en un tableau compatible
    augmented_images_tensor = tf.stack(augmented_images)

    # Prédire sur des échantillons
    predictions = loaded_model.predict(augmented_images_tensor)

    # Calculer les pourcentages basés sur le seuil
    percentage_scores = []
    for pred in predictions:
        if pred < threshold:
            score = (1 - (pred / threshold)) * 100  # Plus proche de 0, pourcentage élevé
        else:
            score = ((pred - threshold) / (1 - threshold)) * 100  # Plus proche de 1, pourcentage élevé
        percentage_scores.append(score)

    # Calculer les classes binaires
    predictions_binary = (predictions > threshold).astype(int).squeeze()
    return predictions, predictions_binary, percentage_scores