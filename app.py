import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Charger le modèle entraîné
model = tf.keras.models.load_model('mnist_cnn_model.h5')




def preprocess_image(image_data):
    """Prétraite l'image dessinée pour la prédiction"""
    if image_data is None:
        return None

    # Gérer le cas où image_data est un dictionnaire (format Sketchpad)
    if isinstance(image_data, dict):
        if 'image' in image_data:
            image = image_data['image']
        elif 'composite' in image_data:
            image = image_data['composite']
        else:
            # Prendre la première clé disponible qui contient une image
            for key, value in image_data.items():
                if isinstance(value, (Image.Image, np.ndarray)):
                    image = value
                    break
            else:
                return None
    else:
        image = image_data

    # Convertir PIL Image en array numpy si nécessaire
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Vérifier que nous avons bien un array numpy
    if not isinstance(image, np.ndarray):
        return None

    # Convertir en niveaux de gris si nécessaire
    if len(image.shape) == 3:
        if image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        elif image.shape[2] == 3:  # RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Redimensionner à 28x28 pixels
    image = cv2.resize(image, (28, 28))

    # Inverser les couleurs (MNIST a des chiffres blancs sur fond noir)
    image = 255 - image

    # Normaliser les valeurs entre 0 et 1
    image = image.astype('float32') / 255.0

    # Ajouter les dimensions pour le batch et le canal
    image = image.reshape(1, 28, 28, 1)

    return image




def predict_digit(image_data):
    """Prédit le chiffre dessiné"""
    try:
        if image_data is None:
            return "Veuillez dessiner un chiffre"

        # Debug: afficher le type de données reçues
        print(f"Type de données reçues: {type(image_data)}")
        if isinstance(image_data, dict):
            print(f"Clés du dictionnaire: {list(image_data.keys())}")

        # Prétraiter l'image
        processed_image = preprocess_image(image_data)

        if processed_image is None:
            return "Erreur lors du prétraitement de l'image"

        # Faire la prédiction
        prediction = model.predict(processed_image, verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Créer le résultat détaillé
        result = f"Chiffre prédit: {predicted_digit}\nConfiance: {confidence:.2f}%\n\n"
        result += "Probabilités pour chaque chiffre:\n"
        for i in range(10):
            prob = prediction[0][i] * 100
            result += f"{i}: {prob:.2f}%\n"

        return result

    except Exception as e:
      return f"Erreur: {str(e)}\nType de données: {type(image_data)}"
  
  
  
  
  
  
# Créer l'interface Gradio
with gr.Blocks(title="Reconnaissance de Chiffres MNIST") as demo:
    gr.Markdown("Reconnaissance de Chiffres MNIST")
    gr.Markdown("Dessinez un chiffre (0-9) dans la zone ci-dessous et cliquez sur 'Prédire' pour voir le résultat!")

    with gr.Row():
        with gr.Column():
            # Zone de dessin - utiliser type="numpy" pour obtenir directement un array
            sketchpad = gr.Sketchpad(
                label="Dessinez un chiffre ici",
                type="numpy",
                canvas_size=(280, 280)
            )

            # Boutons
            with gr.Row():
                predict_btn = gr.Button("Prédire", variant="primary")
                clear_btn = gr.Button("Effacer")

        with gr.Column():
            # Résultats
            result_text = gr.Textbox(
                label="Résultat de la prédiction",
                lines=15,
                interactive=False
            )

    # Fonction pour effacer le canvas
    def clear_canvas():
        return None, ""

    # Événements
    predict_btn.click(
        fn=predict_digit,
        inputs=[sketchpad],
        outputs=[result_text]
    )

    clear_btn.click(
        fn=clear_canvas,
        outputs=[sketchpad, result_text]
    )

if __name__ == "__main__":
    demo.launch()