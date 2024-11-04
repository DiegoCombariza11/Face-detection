import cv2
from deepface import DeepFace

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
    # Detecta y obtiene el embedding usando FaceNet preentrenado

    #Tomar la foto de la cámara
    if cv2.waitKey(1) & 0xFF == ord('t'):
        embedding = DeepFace.represent(frame, model_name="Facenet")
        print(embedding)
        break


# Aquí puedes almacenar el embedding o compararlo con otros en tu base de datos


# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()