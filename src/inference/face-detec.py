import cv2
from deepface import DeepFace
import json
import os
import numpy as np
from scipy.spatial.distance import cosine, euclidean

# Primero cargamos el modelo para que no tarde tanto
model_name = "Facenet"

# Cargamos el json con las caras
with open("face_data.json", "r") as json_file:
    embeddings = json.load(json_file)


# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Cargar el modelo preentrenado para detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Captura de video en tiempo real
    ret, frame = cap.read()

    # Convertir a escala de grises (mejora la precisión)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Dibujar un rectángulo alrededor del rostro detectado
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Mostrar la imagen
    cv2.imshow('Detección de Rostros', frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Presionar 'r' para capturar y reconocer el rostro
    if cv2.waitKey(1) & 0xFF == ord('r'):
        print("Reconociendo rostro...")
        embedding_real_time = DeepFace.represent(frame, model_name=model_name)[0]["embedding"]

        # Convertir el embedding en un vector numpy
        embedding_real_time = np.array(embedding_real_time).flatten()

        # Comparar el embedding con los embeddings almacenados
        recognized_name = "Desconocido"  # Si no hay coincidencia, será "Desconocido"
        min_distance = float("inf")  # Inicializa con un valor alto

        for person in embeddings:
            stored_embedding = np.array(person["embedding"])
            # Calcula la distancia coseno o euclidiana
            distance = cosine(embedding_real_time, stored_embedding)  # Distancia de coseno
            # Alternativamente, puedes usar la distancia euclidiana:
            # distance = euclidean(embedding_real_time, stored_embedding)

            # Si la distancia es menor al umbral, es una coincidencia
            if distance < 0.5:  # Ajusta el umbral según la precisión deseada
                recognized_name = person["name"]
                min_distance = distance
                break  # Deja de buscar si ya encontró una coincidencia

        print(f"Persona reconocida: {recognized_name} (Distancia: {min_distance})")

    # Tomar la foto de la cámara con la tecla 't'
    if cv2.waitKey(1) & 0xFF == ord('t'):
        print("Tomando foto...")
        embedding = DeepFace.represent(frame, model_name=model_name)[0]["embedding"]
        #embedding = DeepFace.represent(frame, model_name=model_name)

        print(embedding)

        # Ingresar el nombre para guardar el archivo
        name = input("Ingrese el nombre de la persona: ")

        # Creamos nuestro formato
        face_data = {
            "name": name,
            "embedding": embedding
        }

        json_path = "face_data.json"

        if os.path.exists(json_path):
            with open(json_path, "r") as json_file:
                data = json.load(json_file)
            data.append(face_data)
        else:
            data = []
            data.append(face_data)

        # Guardamos el json
        with open(json_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        # Salimos del bucle
      

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
