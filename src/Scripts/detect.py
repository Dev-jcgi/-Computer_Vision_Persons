#librerias
import torch
import cv2
import numpy as np

#Leer modelo
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/JC/Documents/YOLO/Proyecto_Final/environment/Scripts/model/model1.pt')

#Captura de imagen
cap = cv2.VideoCapture(0)

#inicio de captura
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el frame de la cámara")
        continue  # Salta al siguiente ciclo del bucle si no hay frame
    #deteccion
    detect = model(frame)

    info = detect.pandas().xyxy[0]
    print (info)

    #show fps
    cv2.imshow('Detecotr de Carros', np.squeeze(detect.render()))

    #iniciar al presionar tecla
    t= cv2.waitKey(5)
    if(t==27):
        break

cap.release()
cv2.destroAllWindows()
