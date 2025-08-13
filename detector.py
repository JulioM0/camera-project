import cv2
import numpy as np

video = cv2.VideoCapture(0) 
fondo = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=True) # sustractor de fondo que sirve para detectar movimiento
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))                               
area = np.array([[70, 60], [590, 60], [590, 450], [70, 450]])                 

direccion_actual = "" 
estaba_dentro = False

while True:
    ret, frame = video.read()
    if ret == False : break

    # crea el segundo frame negro0 que muestra solo los pixeles del objeto dentro del rectsangulo
    mascara = np.zeros(shape=(frame.shape[:2]), dtype = np.uint8) 
    cv2.drawContours(mascara, [area], -1, 255, -1)  
    imagen_area = cv2.bitwise_and(frame, frame, mask=mascara) 

    # detecta el movimiento del objeto
    fg = fondo.apply(imagen_area)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    fg = cv2.dilate(fg, None, iterations=5)
    cv2.imshow('fg', fg)
    contornos, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contornos:
        contorno_mayor = max(contornos, key=cv2.contourArea)
        if cv2.contourArea(contorno_mayor) > 2500:
            x, y, w, h = cv2.boundingRect(contorno_mayor)
            centro_x = x + w // 2
            centro_y = y + h // 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.circle(frame, (centro_x, centro_y), 2, (0, 0, 255), -1)

            # ver si el objeto (todo el rectngulo) esta dentro del limite
            dentro = (x > 70 and x + w < 590)
            if estaba_dentro and not dentro:
                if x + w >= 590:
                    direccion_actual = "Salida"
                    cv2.line(frame, (590, 60), (590, 450), (0, 255, 0), 7) # el color es (0, 255, 0) y el grosor es el ultimo numero
                elif x <= 70:
                    direccion_actual = "Entrada"
                    cv2.line(frame, (70, 60), (70, 450), (0, 255, 0), 7)
            estaba_dentro = dentro

    # dibuja las zonas y muestra texto
    cv2.drawContours(frame, [area], -1, (255, 0, 255), 2)
    cv2.line(frame, (590, 60), (590, 450), (0, 255, 255), 2)
    cv2.line(frame, (70, 60), (70, 450), (0, 255, 255), 2)
    cv2.putText(frame, f"Direccion: {direccion_actual}", (20, 40), cv2.FONT_ITALIC, 0.7, (0, 0, 255), 2)
    
    cv2.imshow('prueba detector de direccion', frame)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()