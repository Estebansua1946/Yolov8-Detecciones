from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np

#cap =cv2.VideoCapture('videos/road.mp4') # cargar video
cap =cv2.VideoCapture('videos/carros.mp4') # cargar video

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

mask = cv2.imread('imagenes/prueba2.png') #cargamos la mascara para delimitar una zona de interes


while True:
    
    success, frame =cap.read()
    zona= cv2.bitwise_and(frame,mask) #se crea la zona de interes
  
    result = modelo(zona,stream = True) 
    
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
                    cvzone.cornerRect(frame,(x1,y1,w,h)) #dibujar rectangulo con cvzone
                    cvzone.putTextRect(frame, f'{classNames[cls]} {conf}',(max(0,x1), max(35,y1)),scale= 0.7, thickness=1) #imprimimos en pantalla el valor de confianza
            #cvzone.putTextRect(frame, f'{cls} {conf}',(max(0,x1), max(35,y1)),scale= 0.7, thickness=1)
            #print(conf)
        
    cv2.imshow("video",frame)
    #cv2.imshow("region",zona)
    cv2.waitKey(1)