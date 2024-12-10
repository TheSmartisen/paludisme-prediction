FROM python:3.10-slim

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y libpq-dev gcc

# Installer les dépendances Python
WORKDIR /app
COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application
COPY . .

# Exposer le port Flask par défaut
EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]