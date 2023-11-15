from flask import Flask, render_template, Response, request, jsonify # importamos la clase Flask y render_template para renderizar los templates de html
import cv2 # importamos opencv
import dlib # importamos dlib
import numpy as np
# import movocular # importamos el modulo movocular
import time # importamos el modulo time

app=Flask(__name__)
#Asignacion de la variable cap para la captura de video

cap = None
data_iris = [] # lista para almacenar los datos de la deteccion de iris
session_timer = None # variable para almacenar el tiempo de sesion

def iniciar_camara(): # funcion para iniciar la camara
    global cap # se declara la variable global cap
    cap = cv2.VideoCapture(0) # se asigna la camara predeterminada a la variable cap


def shape_to_np(shape, dtype="int"): # funcion para convertir los puntos de referencia faciales en una lista de coordenadas (x, y)
    # Inicializar la lista de coordenadas (x, y)
    coords = np.zeros((68, 2), dtype=dtype)
    # Recorrer los 68 puntos de referencia faciales y convertirlos en una tupla de coordenadas (x, y)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # Devolver la lista de coordenadas (x, y)
    return coords

def eye_on_mask(mask,shape, side): # funcion para dibujar los ojos en la mascara
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

def contouring(thresh, mid, img, right=False): # funcion para detectar los contornos de los ojos
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        # Detectar la dirección en la que se está mirando
        direction = get_gaze_direction(cx, cy, mid, img.shape[1])
        # Mostrar el texto en la parte superior izquierda de la ventana
        
        cv2.putText(img, "Direction: {}".format(direction), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    except:
        pass

def get_gaze_direction(x, y, mid, width): # funcion para determinar la direccion de la mirada
    # Calcular la posición relativa en el eje x con respecto al centro
    relative_x = x - mid
    # Calcular el porcentaje de desplazamiento en el eje x
    percentage = (relative_x / mid) * 100
    # Determinar la dirección basándose en el porcentaje
    if percentage > 10:
        return "Derecha"
    elif percentage < -10:
        return "Izquierda"
    else:
        return "Centro"
# Inicializar el detector de caras y el predictor de puntos de referencia faciales
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') 

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]


def generate(): # funcion que genera el video
    iniciar_camara() # se ejecuta la funcion para iniciar la camara
    while True:
        ret, frame = cap.read() # lee los frames capturados por la camara web
        if not ret: # si no hay frames
            break 
        else: 
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detectar caras en la imagen en escala de grises
            faces = detector(gray)

            for face in faces:
                # Obtener los puntos de referencia faciales para la cara actual
                shape = predictor(gray, face)
                shape = shape_to_np(shape)

                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                mask = eye_on_mask(mask,shape, left)
                mask = eye_on_mask(mask,shape, right)
                mask = cv2.dilate(mask, None, iterations=5)

                eyes = cv2.bitwise_and(frame, frame, mask=mask)
                mask = (eyes == [0, 0, 0]).all(axis=2)
                eyes[mask] = [255, 255, 255]

                mid = (shape[42][0] + shape[39][0]) // 2
                eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)

                _, thresh = cv2.threshold(eyes_gray, 75, 255, cv2.THRESH_BINARY)
                thresh = cv2.erode(thresh, None, iterations=2)
                thresh = cv2.dilate(thresh, None, iterations=4)
                thresh = cv2.medianBlur(thresh, 3)
                thresh = cv2.bitwise_not(thresh)

                contouring(thresh[:, 0:mid], mid, frame)
                contouring(thresh[:, mid:], mid, frame, True)

            
            suc, encodedImage = cv2.imencode('.jpg', frame) # codifica los frames en formato jpeg
            frame = encodedImage.tobytes() # convierte los frames codificados en bytes
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              frame + b'\r\n') # retorna los frames codificados en formato jpeg




@app.route('/') # ruta de la pagina principal
def index(): # funcion que se ejecuta cuando se accede a la ruta
    datos = {
        'title': 'Diagnostico'

    }
    return render_template('index.html', data = datos) # renderiza el template index.html

# ruta para el video
@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame") 

@app.route('/detect_eyes') # ruta para detectar los ojos
def detect_eyes():
    datos = {
        'title': 'Movocular',
        'message': 'Diagnostico'

    }
    return render_template('camara.html', data= datos) # renderiza el template camara.html

def iniciar_sesion():
    global session_timer, data_iris # se declaran las variables globales
    iniciar_camara() # se ejecuta la funcion para iniciar la camara
    session_timer = time.time() # se asigna el tiempo actual a la variable session_timer

#metodo que detiene la captura de video y devuelve los datos de la prueba 
@app.route('/detener_sesion', methods = ['GET', 'POST']) # ruta para detener la sesion
def detener_sesion():
    global session_timer, data_iris # se declaran las variables globales
    cap.release()
    session_timer = None
    return jsonify(data_iris)


if __name__ == '__main__': # si el archivo es ejecutado directamente
    app.run(debug=True, port = 5000) # se ejecuta la aplicacion en el puerto 5000 en modo debug 