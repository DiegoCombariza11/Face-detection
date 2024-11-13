# Importación de librerías necesarias
import cv2  # OpenCV para captura de video
import pytesseract  # Tesseract OCR para reconocimiento de texto en la imagen
import pandas as pd  # Pandas para manejar el archivo Excel con la información de las placas
import numpy as np  # Para cálculos y manipulación de datos
import time  # Para pausas en la visualización de resultados
import matplotlib.pyplot as plt  # Para gráficos y visualización de métricas
import seaborn as sns  # Para gráficos más detallados de la matriz de confusión
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve  # Métricas de evaluación

# Configuración de Tesseract OCR (ajusta la ruta si estás en Windows)
# pytesseract.pytesseract.tesseract_cmd = r'path_to_tesseract'

# Cargar la base de datos de Excel con información de placas, propietarios y marcas
try:
    df = pd.read_excel("informacion_vehiculos.xlsx")  # Lee el archivo Excel
except FileNotFoundError:
    print("Archivo Excel no encontrado. Asegúrate de que 'informacion_vehiculos.xlsx' esté en la carpeta.")
    exit()  # Salida del programa si no se encuentra el archivo

# Lista para almacenar los resultados de las pruebas
resultados = []

# Función para buscar información del propietario en la base de datos
def get_vehicle_info(plate_text):
    """
    Dado el texto de una placa, retorna la información del propietario y marca del vehículo.
    """
    # Filtra el DataFrame para encontrar la fila donde la columna 'Placa' coincide con el texto de la placa detectada
    info = df[df['Placa'] == plate_text.strip()]
    if not info.empty:
        # Si se encuentra la placa, extrae el propietario y marca
        propietario = info.iloc[0]['Propietario']
        marca = info.iloc[0]['Marca']
        return f"Propietario: {propietario}, Marca: {marca}"
    else:
        # Si no se encuentra la placa en la base de datos, devuelve un mensaje de no encontrado
        return "Información no encontrada en la base de datos."

# Función para reconocer texto de placa en la imagen capturada
def recognize_plate_text(image):
    """
    Realiza el reconocimiento de texto en la imagen de una placa vehicular.
    """
    # Realiza OCR en la imagen para detectar texto de la placa
    plate_text = pytesseract.image_to_string(image, config='--psm 8')
    plate_text = plate_text.strip()
    return plate_text

# Función principal para capturar video desde la cámara y procesar la placa
def capture_and_recognize_plate(true_plate=None):
    """
    Captura video en tiempo real, permite capturar y reconocer placas al presionar 's'.
    Compara la placa detectada con la placa verdadera y almacena el resultado para evaluación.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo acceder a la cámara.")
        return

    print("Presiona 's' para capturar y reconocer la placa, o 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar la imagen.")
            break

        cv2.imshow("Captura de Placa", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            plate_text = recognize_plate_text(thresh)
            print(f"Placa detectada: {plate_text}")

            vehicle_info = get_vehicle_info(plate_text)
            print(vehicle_info)
            cv2.putText(frame, vehicle_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Resultado de Reconocimiento", frame)
            time.sleep(2)

            if true_plate:
                resultado = {"Placa Verdadera": true_plate, "Placa Detectada": plate_text}
                resultados.append(resultado)

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Función para graficar y analizar la matriz de confusión
def plot_confusion_matrix(resultados):
    """
    Genera y visualiza la matriz de confusión para evaluar aciertos y errores en la detección.
    """
    placas_verdaderas = [r["Placa Verdadera"] for r in resultados]
    placas_detectadas = [r["Placa Detectada"] for r in resultados]
    cm = confusion_matrix(placas_verdaderas, placas_detectadas, labels=list(set(placas_verdaderas)))
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=set(placas_verdaderas), yticklabels=set(placas_verdaderas))
    plt.xlabel('Placas Detectadas')
    plt.ylabel('Placas Verdaderas')
    plt.title('Matriz de Confusión del Reconocimiento de Placas')
    plt.show()

# Función para graficar la curva ROC y calcular el AUC
def plot_roc_curve(resultados):
    """
    Genera la curva ROC y calcula el área bajo la curva (AUC).
    """
    y_true = [1 if r["Placa Verdadera"] == r["Placa Detectada"] else 0 for r in resultados]
    y_scores = [1 if r["Placa Detectada"] else 0 for r in resultados]

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC del Reconocimiento de Placas')
    plt.legend(loc="lower right")
    plt.show()

# Función para graficar la curva de precisión-recall
def plot_precision_recall_curve(resultados):
    """
    Genera la curva de Precisión-Recall para visualizar el rendimiento del sistema.
    """
    y_true = [1 if r["Placa Verdadera"] == r["Placa Detectada"] else 0 for r in resultados]
    y_scores = [1 if r["Placa Detectada"] else 0 for r in resultados]

    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', color='blue')
    plt.xlabel('Recall')
    plt.ylabel('Precisión')
    plt.title('Curva de Precisión-Recall del Reconocimiento de Placas')
    plt.show()

# Función para graficar la evolución de precisión y error en el tiempo
def plot_precision_error_over_time(resultados):
    """
    Grafica la evolución de la precisión y tasa de error acumuladas en el tiempo.
    """
    aciertos = np.cumsum([1 if r["Placa Verdadera"] == r["Placa Detectada"] else 0 for r in resultados])
    total = np.arange(1, len(resultados) + 1)
    precision_acumulada = aciertos / total
    error_acumulado = 1 - precision_acumulada

    plt.figure(figsize=(12, 6))
    plt.plot(total, precision_acumulada, label='Precisión Acumulada', color='green')
    plt.plot(total, error_acumulado, label='Error Acumulado', color='red')
    plt.xlabel('Número de Capturas')
    plt.ylabel('Tasa')
    plt.title('Evolución de la Precisión y Error en el Tiempo')
    plt.legend()
    plt.show()

# Ejemplo de uso con la placa verdadera
true_plate = "ABC123"  # Cambiar por la placa verdadera de prueba
capture_and_recognize_plate(true_plate=true_plate)

# Evaluación de la efectividad después de capturas
plot_confusion_matrix(resultados)
plot_roc_curve(resultados)
plot_precision_recall_curve(resultados)
plot_precision_error_over_time(resultados)