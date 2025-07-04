import cv2
import streamlit as st
import numpy as np
from datetime import datetime

# Charger le classificateur Haar
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_cascade.empty():
    st.error("Erreur : le fichier Haar n'a pas été chargé.")
else:
    print("Le classificateur a bien été chargé.")

# Fonction principale de détection
def detect_faces(scale_factor, min_neighbors, box_color):
    hex_color = box_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    bgr = (rgb[2], rgb[1], rgb[0])

    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while st.session_state.detection_active:
        ret, frame = cap.read()
        if not ret:
            st.error("Erreur de capture vidéo.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

        # Enregistrer une image si demandé
        if st.session_state.save_image:
            filename = f"faces_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            st.success(f"Image enregistrée : {filename}")
            st.session_state.save_image = False

    cap.release()
    cv2.destroyAllWindows()


def app():
    st.title("🧠 Détection de visages avec l'algorithme de Viola-Jones")

    st.markdown("""
    ### Instructions :
    - Cliquez sur **"Démarrer la détection de visages"** pour activer la webcam.
    - Utilisez les **curseurs** pour ajuster la précision de la détection.
    - Choisissez une **couleur** pour les rectangles autour des visages.
    - Cliquez sur **📸 "Enregistrer"** pour sauvegarder une image avec visages détectés.
    - Cliquez sur **❌ "Arrêter"** pour fermer la webcam.
    """)

    # Initialiser les flags si ce n'est pas déjà fait
    if "detection_active" not in st.session_state:
        st.session_state.detection_active = False
    if "save_image" not in st.session_state:
        st.session_state.save_image = False

    scale_factor = st.slider("🔍 scaleFactor (zoom)", 1.05, 2.0, 1.3, 0.05)
    min_neighbors = st.slider("👥 minNeighbors (sensibilité)", 1, 10, 5, 1)
    box_color = st.color_picker("🎨 Couleur du rectangle", "#00FF00")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🎬 Démarrer la détection"):
            st.session_state.detection_active = True

    with col2:
        if st.button("📸 Enregistrer l'image"):
            st.session_state.save_image = True

    with col3:
        if st.button("❌ Arrêter la détection"):
            st.session_state.detection_active = False

    if st.session_state.detection_active:
        detect_faces(scale_factor, min_neighbors, box_color)

if __name__ == "__main__":
    app()
