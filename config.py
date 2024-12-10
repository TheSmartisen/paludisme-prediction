import os 

class Config:
    LABEL_CLASS = ["Parasitized", "Uninfected"]
    MODEL_DIR_PATH = os.path.join("app", "ia-model")
    MODEL_FILE = os.path.join(MODEL_DIR_PATH, "malaria_model.keras")
    METADATA_FILE = os.path.join(MODEL_DIR_PATH, "metadata.txt")
    BASE_DATA_PATH = "/data"
    FEEDBACK_TYPES = ["Correct", "Incorrect"]