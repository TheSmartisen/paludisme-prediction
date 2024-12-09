const dropZone = document.getElementById('dropZone');
const imageInput = document.getElementById('imageUpload');
const resultModal = new bootstrap.Modal(document.getElementById('resultModal'));
const selectedImage = document.getElementById('selectedImage');
const predictionResult = document.getElementById('predictionResult');
const loader = document.getElementById('loader');

// Gérer l'image chargée
imageInput.addEventListener('change', () => {
    const file = imageInput.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            selectedImage.src = e.target.result;
        };
        reader.readAsDataURL(file);

        // Afficher le loader et envoyer l'image à l'API
        loader.style.display = 'block';
        sendImageToApi(file);
    }
});

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        imageInput.files = files;
        const event = new Event('change');
        imageInput.dispatchEvent(event);
    }
});

async function sendImageToApi(file) {
    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch('/api/v1/predict-malaria', {
            method: 'POST',
            body: formData,
        });

        if (response.ok) {
            const data = await response.json();
            const label = data.prediction_label;
            const probability = data.prediction_score.toFixed(2);

            // Appliquer les couleurs selon le label
            if (label === 'Parasitized') {
                predictionResult.innerHTML = `<strong class="prediction-positive">Infecté (Malaria) - ${probability}%</strong>`;
            } else {
                predictionResult.innerHTML = `<strong class="prediction-negative">Non Infecté - ${probability}%</strong>`;
            }

            // Ajouter les événements pour les boutons de feedback
            thumbsUp.onclick = () => {
                sendFeedback(label, true);
                resultModal.hide();
            }
            thumbsDown.onclick = () => {
                sendFeedback(label, false);
                resultModal.hide();
            }
            
            resultModal.show();
        } else {
            alert('Erreur lors de la prédiction. Veuillez réessayer.');
        }
    } catch (error) {
        alert(`Erreur réseau : ${error.message}`);
    } finally {
        loader.style.display = 'none';
    }
}

async function sendFeedback(label, isCorrect) {

    try {
        const feedback = {
            label: label,
            correct: isCorrect || false,
            image: selectedImage.src
        }

        const response = await fetch('/api/v1/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(feedback),
        });

        if (response.ok) {
            alert('Feedback enregistré avec succès. Merci !');
        } else {
            alert("Erreur lors de l'enregistrement du feedback.");
        }

    } catch (error) {
        alert(`Erreur réseau : ${error.message}`);
    }
}