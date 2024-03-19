from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from sort import *


cap =cv2.VideoCapture('videos/carros_720.mp4') # cargar video

modelo = YOLO('yolo weigths/yolov8n.pt')  # modelo de deteccion de objteos

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "girafe", "backpack", "umbrella",
              "handbag",
              "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball", "kite", "baseball bat",
              "basball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottleplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "key borad", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread('imagenes/mascara720.png') #cargamos la mascara para delimitar una zona de interes

#llamamos el rastreador 
rastreador = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limite = [300,370,725,370]
conteo = []

while True:
    
    success, frame =cap.read()
    zona= cv2.bitwise_and(frame,mask) #se crea la zona de interes
  
    result = modelo(zona,stream = True)

    cv2.line(frame,(limite[0], limite[1]), (limite[2], limite[3]),(0, 0, 255),4)

    #lista de detecciones
    detecciones = np.empty((0, 5)) 
    
    #encontramos los puntos x, y , alto y ancho de las detecciones 
    for r in result:
        boxes =r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0] #valores en flotante 
            x1,y1,x2,y2 =  int(x1), int(y1), int(x2), int(y2) #convertimos a entero
            #cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),3) #dibujar rectangulo con opencv

            w,h = x2-x1, y2-y1 # sacamos alto y ancho
            
            #encontramos el valor de confianza entre más cerca al 1 mas certeza de la detección.
            conf = math.ceil((box.conf[0]*100))/100 
            cls = int(box.cls[0])

            if classNames[cls] == "car" and conf > 0.3:
                    #cvzone.cornerRect(frame,(x1,y1,w,h),l =9, t =3) #dibujar rectangulo con cvzone
                    #cvzone.putTextRect(frame, f'{classNames[cls]} {conf}',(max(0,x1), max(35,y1)),scale= 0.5, thickness=1, offset = 3) #imprimimos en pantalla el valor de confianza
                    currentArray =np.array([x1, y1, x2, y2, conf]) # matriz actual de deteccion
                    detecciones = np.vstack((detecciones, currentArray)) # añadimos deteccion a la lista de detecciones
            

    resultados_rastreador = rastreador.update(detecciones)

    #Recorremos la lista de resultado rastreador para obtener los ids
    for resultado in resultados_rastreador:
         
         x1, y1, x2, y2, Id = resultado
         x1,y1,x2,y2 =  int(x1), int(y1), int(x2), int(y2) #convertimos a entero
         w,h = x2-x1, y2-y1

         cvzone.cornerRect(frame,(x1,y1,w,h),l =9, t =3, colorR=(255,0,0)) 
         cvzone.putTextRect(frame, f'{int(Id)}',(max(0,x1), max(35,y1)),scale= 1, thickness=1, offset = 5) #imprimos ID 
         #print(resultado)

         cx, cy = x1+w//2, y1+h//2
         cv2.circle(frame,(cx,cy),4 ,(255,0,255),cv2.FILLED)

         if limite[0] < cx < limite [2] and limite[1]-20 < cy < limite [1]+ 20: 
                         
              if conteo.count(Id) == 0:
                conteo.append(Id)             
    
    cvzone.putTextRect(frame, f'Contador{len(conteo)}',(50,50))      
               

    cv2.imshow("video",frame)
    #cv2.imshow("region",zona)
    cv2.waitKey(1)