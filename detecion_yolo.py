from ultralytics import YOLO
import cv2

modelo = YOLO('../yolo weigths/yolov8l.pt')

result =modelo("imagenes/2.jpg",show = True)
cv2.waitKey(0)