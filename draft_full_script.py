# %%
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

IMG_SIZE = (128, 128)
IMG_SIZE_GREY = (128, 128, 1)

# %%
# Charger le jeu de données Malaria
dataset, info = tfds.load('malaria', split='train', with_info=True, as_supervised=True)

# %%
info

# %%
# Liste pour stocker les dimensions
dimensions = []

# Parcourir le dataset pour collecter les tailles
for image, label in dataset.take(-1):  # Prend toutes les images du dataset
    dimensions.append(image.shape[:2])  # (hauteur, largeur)

# Convertir en numpy array pour faciliter les calculs
dimensions = np.array(dimensions)

# Calculer les statistiques
min_height, min_width = dimensions.min(axis=0)
max_height, max_width = dimensions.max(axis=0)
mean_height, mean_width = dimensions.mean(axis=0)
median_height, median_width = np.median(dimensions, axis=0)

print(f"Dimensions minimales : {min_height}x{min_width}")
print(f"Dimensions maximales : {max_height}x{max_width}")
print(f"Dimensions moyennes  : {mean_height:.1f}x{mean_width:.1f}")
print(f"Dimensions médianes  : {median_height}x{median_width}")

# %%
def load_dataset(dataset, image_size=IMG_SIZE):
    # Convertir image_size en Tensor de type int32
    image_size_tensor = tf.constant(image_size, dtype=tf.int32)

    # Diviser le jeu de données en ensembles d'entraînement et de test
    train_dataset = dataset.take(int(info.splits['train'].num_examples * 0.8))
    test_dataset = dataset.skip(int(info.splits['train'].num_examples * 0.8))

    # Convertir les ensembles en listes de numpy arrays
    X_train, y_train = [], []
    for image, label in train_dataset:
        resized_image = tf.image.resize(image, image_size_tensor).numpy()
        X_train.append(resized_image)
        y_train.append(label.numpy())

    X_test, y_test = [], []
    for image, label in test_dataset:
        resized_image = tf.image.resize(image, image_size_tensor).numpy()
        X_test.append(resized_image)
        y_test.append(label.numpy())

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


# %%
# Charger les ensembles d'entraînement et de test
X_train, y_train, X_test, y_test = load_dataset(dataset)

# %%
# Normaliser les images
X_train_normalized = X_train.astype("float32") / 255
X_test_normalized = X_test.astype("float32") / 255

# %%
# Convertir en niveaux de gris
X_train_grey = X_train_normalized.mean(axis=-1, keepdims=True)  # Garde la dimension de canal (1)
X_test_grey = X_test_normalized.mean(axis=-1, keepdims=True)    # Idem pour l'ensemble de test


# %%
# Afficher quelques exemples d'images et leurs étiquettes
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_train_grey[i].squeeze(), cmap="gray" if X_train_grey.shape[-1] == 1 else None)
    plt.title(f"Etat : {"Uninfected" if y_train[i] == 1 else "Parasitized"}")
    plt.axis("off")
plt.show()

# %%
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Créer le modèle séquentiel
model = Sequential()


# Couche 1 : Convolution 2D
# - 32 filtres (détecte 32 motifs différents)
# - Taille du noyau : (3, 3)
# - Fonction d'activation : ReLU (rectified linear unit)
# - Input_shape : La taille des images en entrée (hauteur, largeur, canaux)
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SIZE_GREY))

# Couche 2 : MaxPooling
# - Réduit la taille de l'image en prenant les valeurs maximales dans une fenêtre (2, 2)
model.add(MaxPooling2D((2, 2)))

# Couche 3 : Convolution 2D
# - 64 filtres pour apprendre des motifs plus complexes
# - Taille du noyau : (3, 3)
# - Fonction d'activation : ReLU
model.add(Conv2D(64, (3, 3), activation='relu'))

# Couche 4 : MaxPooling
# - Une autre couche de pooling pour réduire davantage la taille
model.add(MaxPooling2D((2, 2)))

# Couche 5 : Convolution 2D
# - 128 filtres (détecte des motifs encore plus complexes)
# - Taille du noyau : (3, 3)
# - Fonction d'activation : ReLU
model.add(Conv2D(128, (3, 3), activation='relu'))

# Couche 6 : MaxPooling
# - Réduit encore la taille de l'image
model.add(MaxPooling2D((2, 2)))

# Couche 7 : Flatten
# - Transforme les données 2D en un vecteur 1D pour les couches denses
model.add(Flatten())

# Couche 8 : Dense
# - Couche complètement connectée avec 128 neurones
# - Fonction d'activation : ReLU
model.add(Dense(128, activation='relu'))

# Couche 9 : Dropout
# - Pour éviter le surapprentissage, désactive aléatoirement 50% des neurones
model.add(Dropout(0.5))

# Couche 10 : Dense (Sortie)
# - 1 neurone de sortie pour une classification binaire
# - Fonction d'activation : Sigmoid (produit une probabilité entre 0 et 1)
model.add(Dense(1, activation='sigmoid'))

# %%

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Perte pour une classification binaire
    metrics=['accuracy']        # Suivre la précision pendant l'entraînement
)

# %%

model.summary()

# %%
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',    # Surveiller la perte de validation
    patience=5,            # Nombre d'époques sans amélioration avant d'arrêter
    restore_best_weights=True  # Restaurer les poids du meilleur modèle
)

# %%
# Lancer l'entraînement
history = model.fit(
    X_train_grey,              # Ensemble d'entraînement
    y_train,                    # Labels d'entraînement
    epochs=20,                  # Nombre maximum d'époques
    callbacks=[early_stopping],  # Callback pour arrêter tôt
    validation_split=0.1,
    batch_size=32
)

# %%
# Importer Matplotlib pour les graphiques
import matplotlib.pyplot as plt

# Courbe de précision
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Courbe de perte
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %%
from sklearn.metrics import confusion_matrix, classification_report

#A vous de jouer
# Évaluation de la perte et de la précision
test_loss, test_accuracy= model.evaluate(X_test_grey, y_test)

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test_grey)

# Réduire le seuil
threshold = 0.3  # Ajuster le seuil selon vos besoins
y_pred_binary = (y_pred > threshold).astype(int).squeeze()

# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred_binary)
print("Matrice de confusion avec seuil ajusté :")
print(conf_matrix)

# Visualisation des erreurs de classification
errors = np.where(y_pred_binary != y_test)[0]
plt.figure(figsize=(12, 10))
for i, idx in enumerate(errors[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test_grey[idx].squeeze(), cmap="gray")
    plt.title(f"Vrai :  {"Uninfected" if y_test[idx] == 1 else "Parasitized"}, Prédit : {"Uninfected" if y_pred_binary[idx] == 1 else "Parasitized"}")
    plt.axis("off")
plt.show()


# %% [markdown]
# 1. Observation des erreurs
# 
# Faux négatifs (Parasitized prédit comme Uninfected) :
# - Problème critique : Les cellules infectées ne sont pas détectées, ce qui peut avoir des implications graves, car elles passeront inaperçues.
# - Observation visuelle :
#     - Certaines images de cellules parasitées semblent très similaires aux cellules non infectées (peu de caractéristiques visibles).
#     - La détection des parasites peut être difficile si les parasites ne sont pas distinctement visibles.
# 
# Faux positifs (Uninfected prédit comme Parasitized) :
# - Impact moindre : Bien que moins grave, ce type d'erreur entraîne des traitements inutiles ou des fausses alertes.
# - Observation visuelle :
#     - Dans certaines images, des artefacts ou des variations mineures peuvent être confondus avec des parasites.
#     - Les zones floues ou mal définies dans certaines cellules peuvent également induire le modèle en erreur.

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, classification_report


# Définir des seuils à tester
thresholds = np.arange(0.0, 1.1, 0.1)

# Initialiser des listes pour stocker les métriques
precision_list = []
recall_list = []
f1_list = []

# Calculer les métriques pour chaque seuil
for threshold in thresholds:
    y_pred_binary = (y_pred > threshold).astype(int)
    report = classification_report(y_test, y_pred_binary, target_names=["Parasitized", "Uninfected"], output_dict=True)
    precision_list.append(report["Parasitized"]["precision"])
    recall_list.append(report["Parasitized"]["recall"])
    f1_list.append(report["Parasitized"]["f1-score"])

# Tracer les courbes Precision-Recall et F1-score
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision_list, label="Precision", marker="o")
plt.plot(thresholds, recall_list, label="Recall", marker="o")
plt.plot(thresholds, f1_list, label="F1-Score", marker="o")
plt.xlabel("Seuil")
plt.ylabel("Score")
plt.title("Impact du Seuil sur Precision, Recall et F1-Score")
plt.legend()
plt.grid()
plt.show()

# Tracer la courbe ROC
fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
plt.xlabel("Taux de Faux Positifs (FPR)")
plt.ylabel("Taux de Vrais Positifs (TPR)")
plt.title("Courbe ROC")
plt.legend()
plt.grid()
plt.show()

# Affichage du seuil optimal basé sur la courbe ROC
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = roc_thresholds[optimal_idx]
print(f"Seuil optimal trouvé à partir de la courbe ROC : {optimal_threshold:.2f}")

# %%
threshold = optimal_threshold

y_pred_binary = (y_pred > threshold).astype(int).squeeze()


