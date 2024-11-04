from deepface import DeepFace
import cv2

# Captura una imagen de rostro
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

# Detecta y obtiene el embedding usando FaceNet preentrenado
embedding = DeepFace.represent(frame, model_name="Facenet")

# Aquí puedes almacenar el embedding o compararlo con otros en tu base de datos
print(embedding)
