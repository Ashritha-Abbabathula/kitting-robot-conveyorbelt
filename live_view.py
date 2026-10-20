import cv2
from ultralytics import YOLO

m = YOLO('models/kitting_best.pt')
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ok, f = cap.read()
    if not ok:
        break
    r = m.predict(f, conf=0.5, imgsz=320, verbose=False, device='cpu')[0]
    cv2.imshow('kitting detector (q to quit)', r.plot())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
