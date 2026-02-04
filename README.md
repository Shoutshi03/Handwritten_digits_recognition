# Hand written digits recognition

## Overview

This project implements a handwritten digits recognition using a convolutionnal neural networks (CNNs) trained on MNIST dataset.
the project include a web interface developped with gradio that allows users to draw a digit and get prediction in reel time.

## Functionalities

- **CNN Model** : Architecture de réseau de neurones convolutionnel optimisée pour la reconnaissance de chiffres
- **Interface interactive** : Zone de dessin pour tracer des chiffres à la main
- **Prédictions en temps réel** : Reconnaissance instantanée avec niveau de confiance
- **Analyse détaillée** : Probabilités pour chaque chiffre (0-9)

## Architecture du Modèle

Le modèle CNN comprend :

- 3 couches de convolution avec activation ReLU
- 2 couches de max pooling
- 1 couche dense cachée (64 neurones)
- 1 couche de sortie avec activation softmax (10 classes)

## Project Files

- `main.ipynb` : Script de pipeline.
- `app.py` : Interface Gradio pour l'application web
- `mnist_cnn_model.h5` : Modèle entraîné sauvegardé
- `README.md` : Documentation du projet

## Installation

### requirements

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

### Model training

### Web Interface

```bash
python app.py
```

## Utilisation de l'Interface

1. **Sélectionner l'outil de dessin** (icône pinceau)
2. **Dessiner un chiffre** de 0 à 9 dans la zone de dessin
3. **Cliquer sur "Prédire"** pour obtenir la reconnaissance
4. **Consulter les résultats** :
   - Chiffre prédit
   - Niveau de confiance en pourcentage
   - Probabilités détaillées pour chaque chiffre

## Performances du Modèle

Le modèle atteint une précision élevée sur le dataset de test MNIST grâce à :

- Architecture CNN adaptée aux images
- Prétraitement approprié des données
- Entraînement sur 60,000 images d'entraînement
- Validation sur 10,000 images de test

## Prétraitement des Images

L'application effectue automatiquement :

- Redimensionnement à 28x28 pixels
- Conversion en niveaux de gris
- Inversion des couleurs (fond noir, chiffre blanc)
- Normalisation des valeurs de pixels (0-1)

## Technologies Utilisées

- **TensorFlow/Keras** : Framework de deep learning
- **Gradio** : Interface web interactive
- **OpenCV** : Traitement d'images
- **NumPy** : Calculs numériques
- **PIL** : Manipulation d'images

## Permanent deploiement

in order to deploy permenently applications, we will use Hugging Face [Hugging Face Spaces](https://huggingface.co/spaces) .

```bash
git add .
git commit -m "Initial commit of MNIST Gradio app"
git push
```

Hugging Face Spaces détectera automatiquement votre application Gradio et la déploiera.

Le lien de votre application sera disponible sur la page de votre Space :

[Handwritten_digits_recognition](https://huggingface.co/spaces/Shoutshi03/Handwritten_digits_recognition)
